from tqdm import trange
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


def load_c4_shard(split):
    """Load one C4 shard using verified existing Arrow caches when necessary."""
    import gzip
    from itertools import islice
    import json
    from pathlib import Path
    from datasets import Dataset, concatenate_datasets, config

    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported C4 split: {split}")
    shard = (
        "en/c4-train.00000-of-01024.json.gz" if split == "train"
        else "en/c4-validation.00000-of-00008.json.gz"
    )
    try:
        return load_dataset("allenai/c4", data_files={split: shard}, split=split)
    except (ValueError, FileNotFoundError, ConnectionError) as original_error:
        cache = Path(config.HF_DATASETS_CACHE) / "allenai___c4"
        candidates = []
        for info_path in cache.glob("*/*/*/dataset_info.json"):
            info = json.loads(info_path.read_text(encoding="utf-8"))
            sources = info.get("download_checksums", {})
            if len(sources) != 1 or not any(
                source.endswith("/" + shard) for source in sources
            ):
                continue
            split_info = info.get("splits", {}).get(split)
            files = sorted(info_path.parent.glob(f"c4-{split}*.arrow"))
            if split_info is not None and files:
                candidates.append((info_path, split_info["num_examples"], files))
        if candidates:
            candidates.sort(key=lambda entry: (
                entry[0].parent.name != "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                str(entry[0]),
            ))
            info_path, expected_rows, files = candidates[0]
            dataset = concatenate_datasets([Dataset.from_file(str(file)) for file in files])
            if len(dataset) != expected_rows:
                raise ValueError(f"Incomplete C4 Arrow cache: {info_path.parent}")
            print(f"C4 {split}: local Arrow cache {info_path.parent}")
            return dataset
        if split == "validation":
            from huggingface_hub import hf_hub_download

            raw_path = hf_hub_download(
                repo_id="allenai/c4", repo_type="dataset", filename=shard,
                revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                local_files_only=True,
            )
            with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                rows = [{"text": json.loads(line)["text"]} for line in islice(handle, 1100)]
            if len(rows) != 1100:
                raise ValueError(f"Incomplete C4 validation prefix: {raw_path}")
            return Dataset.from_list(rows)
        raise FileNotFoundError(f"No verified C4 {split} cache under {cache}") from original_error


@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    nll_running = 0
    tokens_processed = 0
    skipped_batches = 0
    for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        lm_logits = model(inputs).logits
        if not torch.isfinite(lm_logits).all():
            print(f"[PPL Warning] batch {i}: non-finite logits detected, skipping.")
            skipped_batches += 1
            continue
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).float()
        if not torch.isfinite(loss):
            print(f"[PPL Warning] batch {i}: non-finite loss detected, skipping.")
            skipped_batches += 1
            continue
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss.item() + b * nll_running
        tokens_processed += shift_labels.numel()
    if tokens_processed == 0:
        print(f"[PPL Error] all {num_samples} batches skipped (non-finite logits/loss). Returning inf.")
        return float("inf")
    if skipped_batches > 0:
        print(f"[PPL Info] skipped {skipped_batches}/{num_samples} batches due to non-finite values.")
    ppl = torch.tensor(nll_running).exp().item()
    return ppl


def get_wikitext2(tokenizer: AutoTokenizer, sequence_length: int):
    test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset_tok = tokenizer(
        "\n\n".join(test_dataset_raw["text"]), return_tensors="pt"
    ).input_ids
    num_test_sequences = test_dataset_tok.numel() // sequence_length
    test_loader = []
    for i in range(num_test_sequences):
        test_loader.append(
            test_dataset_tok[:, i * sequence_length : (i + 1) * sequence_length]
        )
    return test_loader


def get_c4_eval(tokenizer: AutoTokenizer, sequence_length: int):
    val_datasetraw = load_c4_shard("validation")
    valenc = tokenizer(" ".join(val_datasetraw[:1100]["text"]), return_tensors="pt").input_ids
    num_seqs = min(valenc.numel() // sequence_length, 256)
    valenc = valenc[:, : (num_seqs * sequence_length)]
    test_loader = []
    for i in range(num_seqs):
        test_loader.append(valenc[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader
