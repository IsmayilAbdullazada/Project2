import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths
from utils.decode import decode_batch


# -------------------------------------------------
# Config
# -------------------------------------------------

n_mfcc = 15
BATCH_SIZE = 16
LR = 2e-3
MAX_EPOCHS = 100000
TIME_LIMIT = 20 * 60
eps=1e-8

CHECKPOINT_PATH = Path("best_model.pt")
mfcc_transform = get_mfcc_transform(n_mfcc)


# -------------------------------------------------
# CPU ONLY
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------
# Dataset / Loader
# -------------------------------------------------

train_dataset = CLSPDataset(
    subset="trn",
)

dev_dataset = CLSPDataset(
    subset="dev",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=clsp_collate,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=clsp_collate,
)


# -------------------------------------------------
# Model
# -------------------------------------------------

# Improve this model at modules/model.py!
model = SequenceClassifier(
    num_classes=len(train_dataset.scr_letters),
    feat_dim=n_mfcc,
)

model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


# -------------------------------------------------
# Training
# -------------------------------------------------

def train_epoch():
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        mfcc = mfcc_transform(batch['wavs']).transpose(1, 2)
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / (std + eps)
        logits = model(mfcc.to(device))
        logits = logits.view([-1, logits.shape[-1]])
        loss = criterion(logits, batch['letter_targets'].view([-1]).to(device))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

@torch.no_grad()
def evaluate():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dev_loader:
        mfcc = mfcc_transform(batch['wavs']).transpose(1, 2)
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / (std + eps)
        logits = model(mfcc.to(device))
        logits_flat = logits.view([-1, logits.shape[-1]])
        targets = batch['letter_targets'].view([-1]).to(device)
        loss = criterion(logits_flat, targets)
        total_loss += loss.item()

        logit_lengths = wav_lengths_to_logit_lengths(batch['wav_lengths']).to(device)
        preds = decode_batch(logits, logit_lengths, dev_dataset)
        labels = batch["labels"].to(device)
        if preds is not None:
            correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dev_loader)
    acc = correct / total if total > 0 else 0.0

    return avg_loss, acc


# -------------------------------------------------
# Main Loop (time-limited)
# -------------------------------------------------

best_dev_acc = 0.

start_time = time.time()
epoch = 0

while epoch < MAX_EPOCHS:

    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        print("\nTime limit reached")
        break

    epoch += 1

    train_loss = train_epoch()
    dev_loss, dev_acc = evaluate()

    print(
        f"Epoch {epoch:04d} | "
        f"train_loss={train_loss:.4f} | "
        f"dev_loss={dev_loss:.4f} | "
        f"dev_acc={dev_acc:.4f} | "
        f"time={elapsed/60:.2f}m"
    )

    # ---- Save best model (by DEV LOSS) ----
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dev_loss": dev_loss,
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )

        print("Saved new best model")


print("\nTraining finished.")
print("Best dev accuracy:", best_dev_acc)
