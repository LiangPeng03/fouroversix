from tqdm import trange
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


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
    val_datasetraw = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    valenc = tokenizer(" ".join(val_datasetraw[:1100]["text"]), return_tensors="pt").input_ids
    num_seqs = min(valenc.numel() // sequence_length, 256)
    valenc = valenc[:, : (num_seqs * sequence_length)]
    test_loader = []
    for i in range(num_seqs):
        test_loader.append(valenc[:, i * sequence_length : (i + 1) * sequence_length])
    return test_loader
