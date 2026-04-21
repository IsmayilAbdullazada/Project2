#!/usr/bin/env python
# infer.py
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths
from utils.decode import decode_batch

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=str,
        help="dev, tst",
        default="dev"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default="best_model.pt"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="output.txt"
    )
    return parser.parse_args()

# --------------------------------------------------
# Config
# --------------------------------------------------
n_mfcc = 15
BATCH_SIZE = 16
mfcc_transform = get_mfcc_transform(n_mfcc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main(args):
    # --------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------
    CHECKPOINT_PATH = args.checkpoint_path
    OUTPUT_PATH =args.output_path
    test_dataset = CLSPDataset(subset=args.subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=clsp_collate
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = SequenceClassifier(
        num_classes=len(test_dataset.scr_letters),
        feat_dim=n_mfcc
    )
    model.to(device)

    # load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            # compute normalized MFCCs
            mfcc = mfcc_transform(batch['wavs']).transpose(1, 2)
            mean = mfcc.mean(dim=1, keepdim=True)
            std = mfcc.std(dim=1, keepdim=True)
            mfcc = (mfcc - mean) / (std + 1e-8)

            logits = model(mfcc.to(device))

            logit_lengths = wav_lengths_to_logit_lengths(batch['wav_lengths']).to(device)

            # decode predictions: returns vocab index per sample
            pred_vocab_idx = decode_batch(logits, logit_lengths, test_dataset)  # [B], indices into vocab_tensor

            # convert to actual words
            for idx in pred_vocab_idx:
                all_preds.append(test_dataset.scr_vocab[idx])

    # --------------------------------------------------
    # Write to output.txt
    # --------------------------------------------------
    with open(OUTPUT_PATH, "w") as f:
        for word in all_preds:
            f.write(f"{word}\n")

    print(f"Inference finished. Wrote {len(all_preds)} predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    args = check_argv()
    main(args)