from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from fouroversix import FourOverSixLinear, ModelQuantizationConfig
from fouroversix.model.quantize import QuantizedModule

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .rtn import RTNEvaluatorImpl

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

awq_img = get_image(dependencies=[Dependency.fouroversix, Dependency.awq])


def get_awq_calibration_samples(tokenizer, seed, n_samples=128, seqlen=2048):
    """Sample seeded, document-local C4 windows without writing index caches."""
    from .perplexity_mr_gptq import load_c4_shard

    dataset = load_c4_shard("train")
    rng = random.Random(seed)
    document_ids = list(range(len(dataset)))
    rng.shuffle(document_ids)
    samples = []
    for document_id in document_ids:
        encoded = tokenizer.encode(dataset[document_id]["text"])
        if len(encoded) < seqlen:
            continue
        start = rng.randrange(len(encoded) - seqlen + 1)
        samples.append(torch.tensor(
            [encoded[start:start + seqlen]], dtype=torch.long
        ))
        if len(samples) == n_samples:
            break
    if len(samples) != n_samples:
        raise ValueError(
            f"C4 has only {len(samples)} eligible documents, need {n_samples}"
        )
    tokens = torch.cat(samples, dim=0)
    digest = sha256(tokens.numpy().tobytes()).hexdigest()
    print(f"[AWQ calibration] dataset=c4 seed={seed} "
          f"shape={tuple(tokens.shape)} SHA256={digest}")
    return list(tokens.split(1))


class FourOverSixLinearForAWQ(FourOverSixLinear):
    """
    Drop-in replacement for `FourOverSixLinear` that quantizes the weights and
    activations during AWQ calibration.
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.config.keep_master_weights = True
        self.high_precision = False

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when calibrating with AWQ.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that can optionally be run in high precision. This is used to
        calculate the high-precision output to compare to during the auto-scale process
        in AWQ calibration.
        """

        return (
            F.linear(input, self.weight, self.bias)
            if self.high_precision
            else super().forward(input)
        )


@app.cls(
    image=awq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class AWQEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using AWQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        force_requantize_model: bool = False,
        from_pretrained_kwargs: dict[str, Any],
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
    ) -> AutoModelForCausalLM:
        """Quantize a model using AWQ."""

        import torch
        from awq.quantize import pre_quant
        from awq.quantize.pre_quant import apply_awq, run_awq
        from fouroversix import quantize_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Monkey-patch apply_rotary_pos_emb to fix dtype mismatch:
        # RoPE cos/sin are computed in float32; even after cos.to(dtype=x.dtype),
        # PyTorch's type promotion rules make bfloat16 * float32 = float32,
        # causing query/key to be float32 while value stays bfloat16
        # in scaled_dot_product_attention.
        import transformers.models.llama.modeling_llama as llama_module

        _original_apply_rope = llama_module.apply_rotary_pos_emb

        def _patched_apply_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            q_embed, k_embed = _original_apply_rope(
                q, k, cos, sin, position_ids, unsqueeze_dim
            )
            # Force output dtype to match the original input dtype,
            # preventing silent float32 promotion from bfloat16.
            q_embed = q_embed.to(dtype=q.dtype)
            k_embed = k_embed.to(dtype=k.dtype)
            return q_embed, k_embed

        llama_module.apply_rotary_pos_emb = _patched_apply_rope

        # Replace FourOverSixLinear with FourOverSixLinearForAWQ
        QuantizedModule.register(
            nn.Linear,
            replace_existing_modules_in_registry=True,
        )(FourOverSixLinearForAWQ)

        calibration_seed = getattr(self, "calibration_seed", 42)
        save_path = (
            save_path / "awq_c4_128x2048_v1" / model_name.strip("/")
            / f"seed{calibration_seed}" / quantization_config.__hash__()
        )

        if force_requantize_model or not save_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **from_pretrained_kwargs,
            ).eval()

            quantize_model(model, quantization_config)

            enc = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=from_pretrained_kwargs.get(
                    "trust_remote_code",
                    False,
                ),
            )

            import math

            config = model.config
            hidden = getattr(config, "hidden_size", None)
            intermediate = getattr(config, "intermediate_size", None)
            q_group_size = 128
            if hidden is not None and intermediate is not None:
                for gs in (128, 64, 32, 16, 8, 4, 2, 1):
                    if hidden % gs == 0 and intermediate % gs == 0:
                        q_group_size = gs
                        break
            print(f"[AWQ] Using q_group_size={q_group_size} for hidden={hidden}, intermediate={intermediate}")

            supplied_samples = False

            def load_seeded_samples(data="c4", tokenizer=None,
                                    n_samples=128, block_size=2048):
                nonlocal supplied_samples
                if data != "c4":
                    raise ValueError(f"Unexpected AWQ calibration dataset: {data}")
                supplied_samples = True
                return get_awq_calibration_samples(
                    tokenizer, calibration_seed, n_samples, block_size
                )

            with ExitStack() as calibration:
                from awq.utils import calib_data

                calibration.enter_context(patch.object(calib_data, "get_calib_dataset", load_seeded_samples))
                if hasattr(pre_quant, "get_calib_dataset"):
                    calibration.enter_context(patch.object(pre_quant, "get_calib_dataset", load_seeded_samples))
                awq_results = run_awq(
                    model,
                    enc,
                    w_bit=16,
                    q_config={"q_group_size": q_group_size, "zero_point": False},
                    n_samples=128,
                    seqlen=2048,
                    calib_data="c4",
                )
            if not supplied_samples:
                raise RuntimeError("Installed AWQ did not call the seeded calibration loader.")

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(awq_results, save_path)

        # NOTE: We intentionally do NOT restore the original apply_rotary_pos_emb
        # here because the patch is needed for the subsequent perplexity evaluation
        # as well (the same bfloat16 * float32 dtype promotion happens during
        # inference with the quantized model).

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **from_pretrained_kwargs,
        )

        # Apply AWQ
        awq_results = torch.load(save_path, map_location="cuda")
        apply_awq(model, awq_results)

        # Quantize the model
        quantize_model(model, quantization_config)

        return model
