#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.decode import batched_sequence_log_likelihood, build_vocab_tensor
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths


def compute_classification_metrics(gold, pred):
    n = min(len(gold), len(pred))
    gold = gold[:n]
    pred = pred[:n]

    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    labels = sorted(set(gold) | set(pred))

    tp = Counter()
    fp = Counter()
    fn = Counter()

    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    macro_f1 = 0.0
    for c in labels:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_f1 += f1
    macro_f1 = macro_f1 / len(labels) if labels else 0.0

    confusions = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    misrecognized = Counter(g for g, p in zip(gold, pred) if g != p)

    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "errors": sum(g != p for g, p in zip(gold, pred)),
        "top_confusions": [
            {"gold": g, "pred": p, "count": c}
            for (g, p), c in confusions.most_common(15)
        ],
        "top_misrecognized_words": [
            {"word": w, "count": c} for w, c in misrecognized.most_common(15)
        ],
    }


@torch.no_grad()
def compute_alpha_viterbi_metrics(checkpoint_path, subset, n_mfcc, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CLSPDataset(subset=subset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=clsp_collate,
    )

    model = SequenceClassifier(
        num_classes=len(dataset.scr_letters),
        feat_dim=n_mfcc,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mfcc_transform = get_mfcc_transform(n_mfcc)
    eps = 1e-8

    _, vocab_tensor, lengths = build_vocab_tensor(
        dataset.scr2id,
        dataset.letter2id,
        device=device,
    )

    alpha_correct = 0
    viterbi_correct = 0
    total = 0
    decoder_disagree = 0

    for batch in loader:
        mfcc = mfcc_transform(batch["wavs"]).transpose(1, 2)
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / (std + eps)

        logits = model(mfcc.to(device))
        logit_lengths = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)
        labels = batch["labels"].to(device)

        alpha_scores = batched_sequence_log_likelihood(
            logits,
            vocab_tensor,
            lengths,
            dataset.letter2id,
            logit_lengths,
            use_viterbi=False,
        )
        viterbi_scores = batched_sequence_log_likelihood(
            logits,
            vocab_tensor,
            lengths,
            dataset.letter2id,
            logit_lengths,
            use_viterbi=True,
        )

        alpha_pred = alpha_scores.argmax(dim=1)
        viterbi_pred = viterbi_scores.argmax(dim=1)

        alpha_correct += (alpha_pred == labels).sum().item()
        viterbi_correct += (viterbi_pred == labels).sum().item()
        decoder_disagree += (alpha_pred != viterbi_pred).sum().item()
        total += labels.size(0)

    return {
        "subset": subset,
        "alpha_accuracy": alpha_correct / total if total else 0.0,
        "viterbi_accuracy": viterbi_correct / total if total else 0.0,
        "decoder_disagree_count": int(decoder_disagree),
        "n": int(total),
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="dev", choices=["dev", "tst"])
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("output.txt"))
    parser.add_argument("--gold", type=Path, default=Path("data/clsp.devscr"))
    parser.add_argument("--n-mfcc", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--metrics-json", type=Path, default=Path("report_metrics.json"))
    args = parser.parse_args()

    pred = [
        l.strip()
        for l in args.output.read_text().splitlines()
        if l.strip() and "jhucsp" not in l
    ]
    gold = [
        l.strip()
        for l in args.gold.read_text().splitlines()
        if l.strip() and "jhucsp" not in l
    ]

    cls_metrics = compute_classification_metrics(gold, pred)
    decoder_metrics = compute_alpha_viterbi_metrics(
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        n_mfcc=args.n_mfcc,
        batch_size=args.batch_size,
    )

    report = {
        "output_vs_gold": cls_metrics,
        "alpha_vs_viterbi": decoder_metrics,
    }
    args.metrics_json.write_text(json.dumps(report, indent=2))

    print("Saved metrics to", args.metrics_json)
    print("\nOutput vs Gold")
    print("n:", cls_metrics["n"])
    print("accuracy:", round(cls_metrics["accuracy"], 6))
    print("macro_f1:", round(cls_metrics["macro_f1"], 6))
    print("errors:", cls_metrics["errors"])
    print("\nAlpha vs Viterbi")
    print("device:", decoder_metrics["device"])
    print("alpha_accuracy:", round(decoder_metrics["alpha_accuracy"], 6))
    print("viterbi_accuracy:", round(decoder_metrics["viterbi_accuracy"], 6))
    print("decoder_disagree_count:", decoder_metrics["decoder_disagree_count"])


if __name__ == "__main__":
    main()
