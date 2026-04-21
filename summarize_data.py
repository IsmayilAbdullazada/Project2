#!/usr/bin/env python3
import argparse
import json
import statistics
import wave
from collections import Counter
from pathlib import Path


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0


def safe_max(values):
    return max(values) if values else 0


def read_non_header_lines(path: Path):
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "jhucsp" in line.lower():
            continue
        lines.append(line)
    return lines


def summarize_transcripts(lines):
    counts = Counter(lines)
    return {
        "num_utterances": len(lines),
        "num_unique_words": len(counts),
        "top_words": counts.most_common(10),
    }


def summarize_label_lines(lines):
    token_counts = [len(line.split()) for line in lines]
    return {
        "num_utterances": len(lines),
        "tokens_per_utterance": {
            "min": safe_min(token_counts),
            "max": safe_max(token_counts),
            "mean": round(safe_mean(token_counts), 3),
        },
    }


def summarize_wavs(base_dir: Path, split: str, wav_names):
    wav_dir = base_dir / "wav" / split
    rates = Counter()
    channels = Counter()
    durations = []
    missing = []
    unreadable = []

    for name in wav_names:
        wav_path = wav_dir / name
        if not wav_path.exists():
            missing.append(name)
            continue

        try:
            with wave.open(str(wav_path), "rb") as wf:
                nframes = wf.getnframes()
                rate = wf.getframerate()
                ch = wf.getnchannels()
                duration = (nframes / rate) if rate else 0.0

                durations.append(duration)
                rates[rate] += 1
                channels[ch] += 1
        except Exception as exc:  # pragma: no cover
            unreadable.append(f"{name}: {exc}")

    return {
        "num_listed": len(wav_names),
        "num_missing": len(missing),
        "num_unreadable": len(unreadable),
        "duration_seconds": {
            "min": round(min(durations), 4) if durations else 0.0,
            "max": round(max(durations), 4) if durations else 0.0,
            "mean": round(safe_mean(durations), 4),
            "total": round(sum(durations), 2),
        },
        "sample_rate_distribution": dict(sorted(rates.items())),
        "channel_distribution": dict(sorted(channels.items())),
        "missing_examples": missing[:10],
        "unreadable_examples": unreadable[:10],
    }


def summarize_split(data_dir: Path, split: str):
    scr = read_non_header_lines(data_dir / f"clsp.{split}scr")
    lbl = read_non_header_lines(data_dir / f"clsp.{split}lbls")
    wav = read_non_header_lines(data_dir / f"clsp.{split}wav")

    summary = {
        "files_present": {
            "scr": (data_dir / f"clsp.{split}scr").exists(),
            "lbls": (data_dir / f"clsp.{split}lbls").exists(),
            "wav": (data_dir / f"clsp.{split}wav").exists(),
        },
        "line_counts": {
            "scr": len(scr),
            "lbls": len(lbl),
            "wav": len(wav),
        },
        "consistent_num_rows": len({len(scr), len(lbl), len(wav)}) == 1,
    }

    if scr:
        summary["transcripts"] = summarize_transcripts(scr)
    if lbl:
        summary["labels"] = summarize_label_lines(lbl)
    if wav:
        summary["audio"] = summarize_wavs(data_dir, split, wav)

    return summary


def summarize_json_file(path: Path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "name": path.name,
            "parse_ok": False,
            "error": str(exc),
        }

    out = {
        "name": path.name,
        "parse_ok": True,
        "top_level_type": type(obj).__name__,
    }

    if isinstance(obj, list):
        elem_types = Counter(type(x).__name__ for x in obj)
        out["length"] = len(obj)
        out["element_type_distribution"] = dict(elem_types)
        out["sample"] = obj[:5]

    elif isinstance(obj, dict):
        out["length"] = len(obj)
        keys = list(obj.keys())
        out["sample_keys"] = keys[:5]

        value_types = Counter(type(v).__name__ for v in obj.values())
        out["value_type_distribution"] = dict(value_types)

        if value_types == {"int": len(obj)} or value_types == {"float": len(obj)}:
            values = list(obj.values())
            out["value_stats"] = {
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": round(safe_mean(values), 3),
            }

            if values:
                max_key = max(obj, key=obj.get)
                min_key = min(obj, key=obj.get)
                out["largest_entry"] = {"key": max_key, "value": obj[max_key]}
                out["smallest_entry"] = {"key": min_key, "value": obj[min_key]}

        elif value_types == {"list": len(obj)}:
            lengths = [len(v) for v in obj.values()]
            out["list_length_stats"] = {
                "min": safe_min(lengths),
                "max": safe_max(lengths),
                "mean": round(safe_mean(lengths), 3),
            }
    else:
        out["value"] = obj

    return out


def print_human_summary(summary):
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split in ["trn", "dev", "tst"]:
        s = summary["splits"][split]
        print(f"\n[{split.upper()}]")
        print("  files_present:", s["files_present"])
        print("  line_counts:", s["line_counts"])
        print("  consistent_num_rows:", s["consistent_num_rows"])

        if "transcripts" in s:
            t = s["transcripts"]
            print(
                "  transcripts: "
                f"num_utterances={t['num_utterances']} "
                f"num_unique_words={t['num_unique_words']}"
            )
            print("  top_words:", t["top_words"][:5])

        if "labels" in s:
            l = s["labels"]
            print("  labels tokens_per_utterance:", l["tokens_per_utterance"])

        if "audio" in s:
            a = s["audio"]
            print(
                "  audio: "
                f"listed={a['num_listed']} "
                f"missing={a['num_missing']} "
                f"unreadable={a['num_unreadable']}"
            )
            print("  audio duration_seconds:", a["duration_seconds"])
            print("  sample_rate_distribution:", a["sample_rate_distribution"])

    print("\n" + "=" * 70)
    print("JSON FILE SUMMARY")
    print("=" * 70)
    for item in summary["json_files"]:
        status = "OK" if item.get("parse_ok") else "FAIL"
        print(f"\n{item['name']} [{status}]")
        if not item.get("parse_ok"):
            print("  error:", item.get("error"))
            continue
        print("  top_level_type:", item.get("top_level_type"))
        print("  length:", item.get("length"))
        if "value_type_distribution" in item:
            print("  value_type_distribution:", item["value_type_distribution"])
        if "element_type_distribution" in item:
            print("  element_type_distribution:", item["element_type_distribution"])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize CLSP train/dev/test files and JSON metadata."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data_summary.json"),
        help="Path to write machine-readable summary JSON.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    summary = {
        "data_dir": str(data_dir),
        "splits": {
            "trn": summarize_split(data_dir, "trn"),
            "dev": summarize_split(data_dir, "dev"),
            "tst": summarize_split(data_dir, "tst"),
        },
        "json_files": [
            summarize_json_file(p) for p in sorted(data_dir.glob("*.json"))
        ],
    }

    args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_human_summary(summary)
    print(f"\nSaved JSON summary to: {args.json_output}")


if __name__ == "__main__":
    main()
