from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
import time
import torch
from fouroversix import (
    DataType,
    MatmulBackend,
    ModelQuantizationConfig,
    QuantizeBackend,
    ScaleRule,
)

from ..utils import EvaluationFramework

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from transformers import AutoConfig, AutoModelForCausalLM


class PTQEvaluator(ABC):
    """Base class for post-training quantization evaluators."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,  # noqa: ARG003
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """
        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,  # noqa: ARG003
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """
        return {}

    @abstractmethod
    def quantize_model(self, **kwargs: dict[str, Any]) -> AutoModelForCausalLM:
        """Quantize a model."""

    def evaluate(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        eval_framework: EvaluationFramework,
        force_requantize_model: bool = False,
        limit: int | None,
        max_length: int | None,
        tasks: list[str],
        trust_remote_code: bool = False,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
        activation_dtype: DataType | None = None,
        activation_scale_rule: ScaleRule | None = None,
        scale_rule: ScaleRule | None = None,
        weight_dtype: DataType | None = None,
        weight_scale_2d: bool = False,
        weight_scale_rule: ScaleRule | None = None,
        save_path: Path | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        with torch.inference_mode():
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            quantization_config = ModelQuantizationConfig(
                activation_dtype=activation_dtype,
                activation_scale_rule=activation_scale_rule,
                dtype=dtype,
                matmul_backend=matmul_backend,
                output_dtype=DataType(
                    (
                        str(getattr(model_config, "dtype", None)).replace("torch.", "")
                        if getattr(model_config, "dtype", None) is not None
                        else "bfloat16"
                    ),
                ),
                quantize_backend=quantize_backend,
                scale_rule=scale_rule,
                weight_scale_2d=weight_scale_2d,
                weight_dtype=weight_dtype,
                weight_scale_rule=weight_scale_rule,
            )

            # The CUDA quantize/matmul backends only work reliably with bfloat16
            # on Blackwell GPUs. Even though model_config may say float16 (e.g.
            # Llama-2-7b), loading as float16 triggers "Backend cuda does not
            # support the given parameters" during activation quantization.
            # bfloat16 is safe for all models and has comparable precision.
            _model_torch_dtype = torch.bfloat16

            # When running locally with --gpus-per-worker > 1, device will be a
            # string like "cuda:0,1,2,3", but this is not a valid torch device.
            # Here we create a max_memory dict to pass to from_pretrained and save
            # the first GPU as the main device.
            if "," in device:
                from_pretrained_kwargs = {
                    "device_map": "auto",
                    # Limit the memory usage of the first GPU to 80% of the
                    # device's free memory since we'll be letting lm-eval use this
                    # GPU to load tokens and etc.
                    "max_memory": {
                        i: (
                            torch.cuda.mem_get_info(i)[0]
                            * (0.8 if i == int(device.replace("cuda:", "")[0]) else 1)
                            if str(i) in device
                            else 0
                        )
                        for i in range(torch.cuda.device_count())
                    },
                    "torch_dtype": _model_torch_dtype,
                    "trust_remote_code": trust_remote_code,
                }

                device = device.split(",", maxsplit=1)[0]
            else:
                from_pretrained_kwargs = {
                    "device_map": device,
                    "torch_dtype": _model_torch_dtype,
                    "trust_remote_code": trust_remote_code,
                }

            seed = kwargs.pop("seed", None)
            self.calibration_seed = seed if seed is not None else 42
            if seed is not None:
                import random
                import numpy as np
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            quant_start = time.time()
            model = self.quantize_model(
                model_name=model_name,
                force_requantize_model=force_requantize_model,
                save_path=save_path,
                quantization_config=quantization_config,
                from_pretrained_kwargs=from_pretrained_kwargs,
                **kwargs,
            )
            quant_end = time.time()
            print(f"QUANT_TIME: {quant_end - quant_start:.2f}")

            eval_start = time.time()
            # ---------- Custom PPL tasks (MR-GPTQ style) ----------
            ppl_tasks = [t for t in tasks if t in ("wikitext2", "c4")]
            lm_eval_tasks = [t for t in tasks if t not in ("wikitext2", "c4")]
            results = []

            # Load tokenizer unconditionally so it can be passed to HFLM later
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

            if ppl_tasks:
                from .perplexity_mr_gptq import compute_perplexity, get_wikitext2, get_c4_eval

                model.to(device)
                model.config.use_cache = True

                for ppl_task in ppl_tasks:
                    seq_len = max_length or 2048
                    if ppl_task == "wikitext2":
                        eval_data = get_wikitext2(tokenizer, seq_len)
                        ppl = compute_perplexity(model, eval_data)
                        results.append(
                            (
                                "wikitext2",
                                "perplexity,none",
                                ppl,
                                {"perplexity,none": ppl},
                            )
                        )
                        del eval_data
                    elif ppl_task == "c4":
                        c4_data = get_c4_eval(tokenizer, seq_len)
                        ppl = compute_perplexity(model, c4_data)
                        results.append(
                            (
                                "c4",
                                "perplexity,none",
                                ppl,
                                {"perplexity,none": ppl},
                            )
                        )
                        del c4_data
                    torch.cuda.empty_cache()
            # -------------------------------------------------------

            if eval_framework == EvaluationFramework.lm_eval and lm_eval_tasks:
                from lm_eval import evaluator
                from lm_eval.models.huggingface import HFLM
                from lm_eval.tasks import TaskManager

                _orig_max_pos = model.config.max_position_embeddings
                model.config.max_position_embeddings = min(model.config.max_position_embeddings, 2048)

                # Move model to a single device to avoid device-mismatch errors
                # after quantization (quantized layers may end up on CPU when
                # model was loaded with device_map="auto")
                model.to(device)

                task_configs = {
                    "winogrande": {"num_fewshot": 5, "batch_size": 64},
                    "hellaswag": {"num_fewshot": 10, "batch_size": 8},
                    "piqa": {"num_fewshot": 0, "batch_size": 64},
                    "arc_challenge": {"num_fewshot": 25, "batch_size": 8},
                    "boolq": {"num_fewshot": 0, "batch_size": 8},
                    "gsm8k": {"num_fewshot": 5, "batch_size": 16},
                    "mmlu": {"num_fewshot": 5, "batch_size": 4, "limit": 200},
                    "arc_easy": {"num_fewshot": 25, "batch_size": 8, "limit": 4000},
                    "mbpp": {"num_fewshot": 3, "batch_size": 16},
                    "lambada_openai": {"num_fewshot": 0, "batch_size": 16},
                    "gsm8k_llama": {
                        "num_fewshot": 0,
                        "batch_size": 8,
                        "apply_chat_template": True,
                        "fewshot_as_multiturn": True,
                    },
                    "mmlu_cot_llama": {
                        "num_fewshot": 0,
                        "batch_size": 8,
                        "apply_chat_template": True,
                        "fewshot_as_multiturn": True,
                    },
                }

                for task_name in lm_eval_tasks:
                    cfg = task_configs.get(task_name, {"num_fewshot": 0, "batch_size": 1})
                    batch_size = cfg["batch_size"]
                    num_fewshot = cfg.get("num_fewshot", 0)
                    extra_kwargs = {k: v for k, v in cfg.items() if k not in ("batch_size", "num_fewshot", "limit")}

                    # Use task-specific limit if set, otherwise fall back to defaults
                    task_limit = cfg.get("limit")
                    if task_limit is None:
                        if task_name in ("hellaswag"):
                            task_limit = 4000
                        else:
                            task_limit = limit

                    lm = HFLM(
                        pretrained=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=max_length,
                        batch_size=batch_size,
                    )

                    task_results = evaluator.simple_evaluate(
                        model=lm,
                        tasks=[task_name],
                        device=device,
                        limit=task_limit,
                        num_fewshot=num_fewshot,
                        confirm_run_unsafe_code=True,   
                        task_manager=TaskManager(
                            include_path=(
                                Path(__file__).parent.parent / "tasks"
                            ).as_posix(),
                        ),
                        **extra_kwargs,
                    )

                    result = task_results["results"][task_name]

                    if "acc_norm,none" in result:
                        metric_name = "acc_norm,none"
                    elif "acc,none" in result:
                        metric_name = "acc,none"
                    elif "exact_match,strict-match" in result:
                        metric_name = "exact_match,strict-match"
                    elif "exact_match,none" in result:
                        metric_name = "exact_match,none"
                    elif "pass_at_1,none" in result:
                        metric_name = "pass_at_1,none"
                    elif "pass@1,none" in result:
                        metric_name = "pass@1,none"
                    elif "word_perplexity,none" in result:
                        metric_name = "word_perplexity,none"
                    else:
                        metric_name = None

                    results.append(
                        (
                            task_name,
                            metric_name,
                            result.get(metric_name),
                            result,
                        ),
                    )

                    del lm
                    torch.cuda.empty_cache()

                model.config.max_position_embeddings = _orig_max_pos

            elif eval_framework == EvaluationFramework.inspect_ai and lm_eval_tasks:
                import inspect_ai
                from inspect_ai.model import Model
                from inspect_ai.model._generate_config import GenerateConfig

                from .utils import local_hf

                config = GenerateConfig()
                full_results = inspect_ai.eval(
                    tasks=lm_eval_tasks,
                    model=Model(local_hf(model_name, model, config), config, None),
                    limit=limit,
                    log_dir=(save_path / "inspect_ai_logs").as_posix(),
                    log_realtime=True,
                    display="none",
                )

                for log in full_results:
                    metrics = {
                        k: v.value
                        for score in log.results.scores
                        for k, v in score.metrics.items()
                    }

                    metric_name = "accuracy" if "accuracy" in metrics else None

                    results.append(
                        (
                            log.eval.task,
                            metric_name,
                            metrics.get(metric_name),
                            metrics,
                        ),
                    )

            del model
            torch.cuda.empty_cache()

            eval_end = time.time()
            print(f"EVAL_TIME: {eval_end - eval_start:.2f}")

        return results

    @modal.method()
    def evaluate_on_modal(
        self,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model on Modal."""
        results = self.evaluate(*args, **kwargs)
        print(results)
        return results
