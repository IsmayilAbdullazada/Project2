import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.decode import batched_sequence_log_likelihood, build_vocab_tensor
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths


def edit_distance_is_one(a, b):
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if a == b:
        return False

    if la == lb:
        mismatches = sum(x != y for x, y in zip(a, b))
        return mismatches == 1

    # Handle insertion/deletion case in O(n).
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    i = 0
    j = 0
    used_skip = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if used_skip:
            return False
        used_skip = True
        j += 1

    return True


def distinguishing_letters(gold_word, pred_word):
    if len(gold_word) == len(pred_word):
        idxs = [i for i, (g, p) in enumerate(zip(gold_word, pred_word)) if g != p]
        if len(idxs) == 1:
            i = idxs[0]
            return {gold_word[i]}, {pred_word[i]}

    gold_count = Counter(gold_word)
    pred_count = Counter(pred_word)
    gold_only = set((gold_count - pred_count).elements())
    pred_only = set((pred_count - gold_count).elements())
    return gold_only, pred_only


def summarize_float_list(values):
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "n": 0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    else:
        median = sorted_vals[mid]

    return {
        "mean": float(sum(sorted_vals) / n),
        "median": float(median),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "n": n,
    }


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

    alpha_gap_correct = []
    alpha_gap_incorrect = []

    id2letter = {i: c for i, c in enumerate(dataset.scr_letters)}
    id2word = {i: w.lower() for i, w in enumerate(dataset.scr_vocab)}

    one_letter_errors = 0
    one_letter_gold_present = 0
    one_letter_pred_present = 0
    one_letter_gold_checks = 0
    one_letter_pred_checks = 0
    one_letter_pair_stats = defaultdict(lambda: {"count": 0, "gold_letter_seen": 0, "pred_letter_seen": 0})

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

        top2 = torch.topk(alpha_scores, k=2, dim=1).values
        gaps = (top2[:, 0] - top2[:, 1]).detach().cpu().tolist()

        alpha_pred_cpu = alpha_pred.detach().cpu().tolist()
        labels_cpu = labels.detach().cpu().tolist()
        logit_lengths_cpu = logit_lengths.detach().cpu().tolist()

        for i, gap in enumerate(gaps):
            if alpha_pred_cpu[i] == labels_cpu[i]:
                alpha_gap_correct.append(float(gap))
            else:
                alpha_gap_incorrect.append(float(gap))

            gold_word = id2word.get(labels_cpu[i], "")
            pred_word = id2word.get(alpha_pred_cpu[i], "")
            if alpha_pred_cpu[i] == labels_cpu[i]:
                continue
            if not edit_distance_is_one(gold_word, pred_word):
                continue

            one_letter_errors += 1
            gold_letters, pred_letters = distinguishing_letters(gold_word, pred_word)

            valid_t = int(logit_lengths_cpu[i])
            frame_ids = logits[i, :valid_t].argmax(dim=-1).detach().cpu().tolist()
            frame_letters = [id2letter.get(fid, "") for fid in frame_ids]

            has_gold = any(ch in frame_letters for ch in gold_letters)
            has_pred = any(ch in frame_letters for ch in pred_letters)

            if gold_letters:
                one_letter_gold_checks += 1
                one_letter_gold_present += int(has_gold)
            if pred_letters:
                one_letter_pred_checks += 1
                one_letter_pred_present += int(has_pred)

            pair_key = f"{gold_word}->{pred_word}"
            one_letter_pair_stats[pair_key]["count"] += 1
            one_letter_pair_stats[pair_key]["gold_letter_seen"] += int(has_gold)
            one_letter_pair_stats[pair_key]["pred_letter_seen"] += int(has_pred)

        alpha_correct += (alpha_pred == labels).sum().item()
        viterbi_correct += (viterbi_pred == labels).sum().item()
        decoder_disagree += (alpha_pred != viterbi_pred).sum().item()
        total += labels.size(0)

    pair_rows = []
    for pair, vals in sorted(one_letter_pair_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
        c = vals["count"]
        pair_rows.append(
            {
                "pair": pair,
                "count": c,
                "gold_letter_seen_rate": vals["gold_letter_seen"] / c if c else None,
                "pred_letter_seen_rate": vals["pred_letter_seen"] / c if c else None,
            }
        )

    return {
        "subset": subset,
        "alpha_accuracy": alpha_correct / total if total else 0.0,
        "viterbi_accuracy": viterbi_correct / total if total else 0.0,
        "decoder_disagree_count": int(decoder_disagree),
        "n": int(total),
        "device": str(device),
        "top1_top2_gap_alpha": {
            "correct": summarize_float_list(alpha_gap_correct),
            "incorrect": summarize_float_list(alpha_gap_incorrect),
        },
        "one_letter_error_frame_analysis": {
            "num_one_letter_errors": int(one_letter_errors),
            "gold_distinct_letter_present_rate": (
                one_letter_gold_present / one_letter_gold_checks if one_letter_gold_checks else None
            ),
            "pred_distinct_letter_present_rate": (
                one_letter_pred_present / one_letter_pred_checks if one_letter_pred_checks else None
            ),
            "pair_breakdown": pair_rows[:15],
        },
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
    gap = decoder_metrics["top1_top2_gap_alpha"]
    print("top1-top2 gap (correct, mean):", None if gap["correct"]["mean"] is None else round(gap["correct"]["mean"], 6))
    print("top1-top2 gap (incorrect, mean):", None if gap["incorrect"]["mean"] is None else round(gap["incorrect"]["mean"], 6))
    one_letter = decoder_metrics["one_letter_error_frame_analysis"]
    print("one-letter errors:", one_letter["num_one_letter_errors"])
    print(
        "gold-distinct-letter present rate:",
        None
        if one_letter["gold_distinct_letter_present_rate"] is None
        else round(one_letter["gold_distinct_letter_present_rate"], 6),
    )
    print(
        "pred-distinct-letter present rate:",
        None
        if one_letter["pred_distinct_letter_present_rate"] is None
        else round(one_letter["pred_distinct_letter_present_rate"], 6),
    )


if __name__ == "__main__":
    main()
