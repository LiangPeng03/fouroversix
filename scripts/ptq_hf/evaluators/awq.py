from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import gc
from hashlib import sha256
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

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


def _awq_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _awq_to(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_awq_to(item, device) for item in value)
    if isinstance(value, list):
        return [_awq_to(item, device) for item in value]
    return value


def _awq_forward(module, inputs, kwargs):
    output = module(inputs, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def _awq_batch_kwargs(sample_kwargs, start, stop, device):
    """Combine per-sequence masks and RoPE, preserving shared cache positions."""
    def combine(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            if first.ndim >= 2 and first.shape[0] == 1:
                return torch.cat(values, dim=0).to(device)
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError("AWQ received incompatible shared position tensors")
            return first.to(device)
        if isinstance(first, dict):
            return {key: combine([value[key] for value in values]) for key in first}
        if isinstance(first, tuple):
            return tuple(combine([value[index] for value in values])
                         for index in range(len(first)))
        if isinstance(first, list):
            return [combine([value[index] for value in values])
                    for index in range(len(first))]
        if any(value != first for value in values[1:]):
            raise ValueError("AWQ received incompatible per-sequence kwargs")
        return first
    return combine(sample_kwargs[start:stop])


@contextmanager
def _awq_cached_weights(block):
    """Quantize each fixed candidate weight once for all calibration batches."""
    from fouroversix import quantize

    cached = []
    try:
        for module in block.modules():
            if (isinstance(module, FourOverSixLinearForAWQ)
                    and not module.high_precision
                    and not module.config.disable_fprop_quantization):
                previous = getattr(module, "_quantized_weight", None)
                cached.append((module, previous))
                module._quantized_weight = quantize(
                    module.weight.data, module.config.get_weight_config()
                )
        yield
    finally:
        for module, previous in cached:
            if previous is None:
                if hasattr(module, "_quantized_weight"):
                    del module._quantized_weight
            else:
                module._quantized_weight = previous


@torch.no_grad()
def _search_awq_scale_batched(block, linears, inputs, sample_kwargs, w_bit, q_config,
                              batch_size):
    """Evaluate every AWQ candidate over all sequences using bounded batches."""
    from awq.quantize.quantizer import pseudo_quantize_tensor

    device = next(block.parameters()).device
    free_bytes, _ = torch.cuda.mem_get_info(device)
    cache_budget = int(free_bytes * 0.4)
    input_bytes = inputs.numel() * inputs.element_size()
    cache_on_gpu = (
        os.environ.get("FOUROVERSIX_AWQ_GPU_CACHE", "1") == "1"
        and 2 * input_bytes < cache_budget
    )
    working_inputs = inputs.to(device) if cache_on_gpu else inputs
    cache_used = input_bytes if cache_on_gpu else 0
    channel_sum = torch.zeros(inputs.shape[-1], dtype=torch.float32)
    num_tokens = 0
    references = []
    with _awq_cached_weights(block):
        for start in range(0, inputs.shape[0], batch_size):
            stop = start + batch_size
            sample = inputs[start:stop]
            channel_sum += sample.float().abs().reshape(-1, sample.shape[-1]).sum(0)
            num_tokens += sample.numel() // sample.shape[-1]
            kwargs = _awq_batch_kwargs(sample_kwargs, start, stop, device)
            output = _awq_forward(block, working_inputs[start:stop].to(device), kwargs)
            output_bytes = output.numel() * output.element_size()
            if cache_on_gpu and cache_used + output_bytes < cache_budget:
                reference = output
                cache_used += output_bytes
            else:
                reference = output.cpu()
            references.append((start, reference, kwargs))
            del output
    x_max = (channel_sum / num_tokens).to(device=device, dtype=inputs.dtype)
    state = block.state_dict()
    state_bytes = sum(value.numel() * value.element_size() for value in state.values())
    state_device = device if cache_on_gpu and cache_used + state_bytes < cache_budget else "cpu"
    original_state = {
        key: value.detach().to(state_device).clone() for key, value in state.items()
    }
    best_error = float("inf")
    best_scales = None
    try:
        for candidate in range(20):
            scales = x_max.pow(candidate / 20).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for linear in linears:
                linear.weight.mul_(scales.view(1, -1))
                quantized = (
                    pseudo_quantize_tensor(linear.weight.data, n_bit=w_bit, **q_config)
                    if w_bit is not None else linear.weight.data
                )
                linear.weight.copy_(quantized / scales.view(1, -1))
                del quantized
            squared_error = torch.zeros((), device=device, dtype=torch.float64)
            num_values = 0
            with _awq_cached_weights(block):
                for start, reference, kwargs in references:
                    stop = start + batch_size
                    output = _awq_forward(block, working_inputs[start:stop].to(device), kwargs)
                    error = (reference.to(device) - output).float()
                    squared_error.add_(error.square().sum(dtype=torch.float64))
                    num_values += error.numel()
                    del output, error
            loss = squared_error.item() / num_values
            if loss < best_error:
                best_error = loss
                best_scales = scales.detach().cpu().clone()
            block.load_state_dict(original_state)
    finally:
        block.load_state_dict(original_state)
    if best_scales is None or not torch.isfinite(best_scales).all():
        raise RuntimeError("AWQ scale search produced no finite candidate")
    return best_scales


@torch.no_grad()
def run_awq_batched(model, samples, w_bit, q_config, batch_size=8):
    """Layer-wise Llama/Qwen AWQ with CPU activation caches and bounded batches."""
    from awq.quantize import pre_quant
    from awq.utils.module import append_str_prefix, get_op_name
    from tqdm import tqdm

    if model.config.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError(f"Batched AWQ does not support {model.config.model_type}")
    model.config.use_cache = False
    model.cpu()
    torch.cuda.empty_cache()
    layers = pre_quant.get_blocks(model)
    captured_inputs = []
    sample_kwargs = []

    class CaptureComplete(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inputs, **kwargs):
            captured_inputs.append(inputs.detach().cpu())
            sample_kwargs.append(_awq_to(
                {key: value for key, value in kwargs.items() if key != "use_cache"},
                "cpu",
            ))
            raise CaptureComplete

    original_first = layers[0]
    layers[0] = Catcher(original_first)
    try:
        pre_quant.move_embed(model, "cuda")
        for sample in samples:
            try:
                model(sample.to("cuda"))
            except CaptureComplete:
                pass
    finally:
        layers[0] = original_first
        pre_quant.move_embed(model, "cpu")
    if len(captured_inputs) != len(samples):
        raise RuntimeError("AWQ did not capture every calibration sequence")
    inputs = torch.cat(captured_inputs, dim=0)
    del captured_inputs
    results = {"scale": [], "clip": []}
    print(f"[AWQ batched] sequences={len(samples)} seqlen={samples[0].shape[1]} "
          f"GPU batch={batch_size} candidates=20")

    for layer_index, layer in enumerate(tqdm(layers, desc="Running batched AWQ")):
        torch.cuda.reset_peak_memory_stats()
        layer.cuda()
        features = defaultdict(list)
        handles = []
        next_inputs = []

        def make_hook(name):
            def hook(module, args, output):
                features[name].append(args[0].detach().cpu())
            return hook

        try:
            for name, linear in pre_quant.get_named_linears(layer).items():
                handles.append(linear.register_forward_hook(make_hook(name)))
            for start in range(0, inputs.shape[0], batch_size):
                stop = start + batch_size
                output = _awq_forward(
                    layer, inputs[start:stop].to("cuda"),
                    _awq_batch_kwargs(sample_kwargs, start, stop, "cuda"),
                )
                next_inputs.append(output.cpu())
                del output
        finally:
            for handle in handles:
                handle.remove()
        inputs = torch.cat(next_inputs, dim=0)
        del next_inputs, handles
        features = {name: torch.cat(parts, dim=0) for name, parts in features.items()}
        empty_kwargs = [{} for _ in sample_kwargs]
        specs = [
            (layer.input_layernorm,
             [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
             "self_attn.q_proj", layer.self_attn, sample_kwargs),
        ]
        if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
            specs.append((layer.self_attn.v_proj, [layer.self_attn.o_proj],
                          "self_attn.o_proj", layer.self_attn.o_proj, empty_kwargs))
        specs.extend([
            (layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj],
             "mlp.gate_proj", layer.mlp, empty_kwargs),
            (layer.mlp.up_proj, [layer.mlp.down_proj],
             "mlp.down_proj", layer.mlp.down_proj, empty_kwargs),
        ])
        scales_list = []
        for previous, linears, feature_name, inspected, kwargs in specs:
            scales = _search_awq_scale_batched(
                inspected, linears, features[feature_name], kwargs, w_bit, q_config,
                batch_size,
            )
            scales_list.append((
                get_op_name(layer, previous),
                tuple(get_op_name(layer, linear) for linear in linears),
                scales,
            ))
        pre_quant.apply_scale(layer, scales_list, features)
        prefix = get_op_name(model, layer) + "."
        results["scale"] += append_str_prefix(scales_list, prefix)
        clip_list = pre_quant.auto_clip_block(
            layer, w_bit=w_bit, q_config=q_config, input_feat=features
        )
        pre_quant.apply_clip(layer, clip_list)
        results["clip"] += append_str_prefix(clip_list, prefix)
        print(f"[AWQ memory] layer={layer_index} "
              f"peak_allocated_GiB={torch.cuda.max_memory_allocated() / 2**30:.2f}")
        layer.cpu()
        del features, specs, scales_list, clip_list
        gc.collect()
        torch.cuda.empty_cache()
    return results


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
        from awq.quantize.pre_quant import apply_awq
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
        smoke_test = os.environ.get("FOUROVERSIX_SMOKE_TEST") == "1"
        sample_count = 2 if smoke_test else 128
        sequence_length = 512 if smoke_test else 2048
        batch_size = int(os.environ.get("FOUROVERSIX_AWQ_BATCH_SIZE", "8"))
        if batch_size < 1:
            raise ValueError("FOUROVERSIX_AWQ_BATCH_SIZE must be positive")
        batch_size = min(batch_size, sample_count)
        if smoke_test:
            print("[SMOKE TEST] AWQ calibration: C4, 2 x 512 tokens")
        save_path = (
            save_path / f"awq_c4_{sample_count}x{sequence_length}_batch{batch_size}_v2" / model_name.strip("/")
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

            samples = get_awq_calibration_samples(
                enc, calibration_seed, sample_count, sequence_length
            )
            awq_results = run_awq_batched(
                model, samples, w_bit=16,
                q_config={"q_group_size": q_group_size, "zero_point": False},
                batch_size=batch_size,
            )

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
