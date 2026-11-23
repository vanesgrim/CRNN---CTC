from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Asegura que los imports "from src..." funcionen al correr "python src/train.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import NUM_CLASSES, PlateDataset, ctc_collate_fn
from src.model import CRNN
from src.utils import evaluate_predictions, greedy_decode


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_ctc_loss(
    criterion: nn.CTCLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # CTCLoss exige (T, N, C) en log-probabilidades
    log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
    input_lengths = torch.full(
        size=(logits.size(0),),
        fill_value=logits.size(1),  # T = 32
        dtype=torch.long,
        device="cpu" if device.type == "mps" else device,
    )

    if device.type == "mps":
        return criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
    return criterion(log_probs, targets.to(device), input_lengths, target_lengths.to(device))


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets, target_lengths, _ in loader:
        images = images.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = compute_ctc_loss(criterion, logits, targets, target_lengths, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validate(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[float, dict[str, float], list[tuple[str, str, float]]]:
    model.eval()
    running_loss = 0.0
    all_preds: list[str] = []
    all_targets: list[str] = []
    ejemplos: list[tuple[str, str, float]] = []

    for images, targets, target_lengths, raw_labels in loader:
        images = images.to(device)
        logits = model(images)
        loss = compute_ctc_loss(criterion, logits, targets, target_lengths, device)
        running_loss += loss.item() * images.size(0)

        decoded = greedy_decode(logits.cpu())
        for real, (pred, conf) in zip(raw_labels, decoded):
            all_targets.append(real)
            all_preds.append(pred)
            if len(ejemplos) < 4:
                ejemplos.append((real, pred, conf))

    avg_loss = running_loss / max(1, len(loader.dataset))
    metrics = evaluate_predictions(all_preds, all_targets)
    return avg_loss, metrics, ejemplos


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento CRNN-CTC para lectura de placas")
    parser.add_argument("--train-dir", default="data/raw/imgs/train", help="Carpeta de placas train")
    parser.add_argument("--val-dir", default="data/raw/imgs/val", help="Carpeta de placas val")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas (doc: 50-100)")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote (doc: 64-128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje inicial")
    parser.add_argument("--out", default="checkpoints/crnn_best.pt", help="Ruta del mejor modelo")
    parser.add_argument("--no-augment", action="store_true", help="Desactiva aumentos en train")
    args = parser.parse_args()

    device = get_device()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = PlateDataset(args.train_dir, augment=not args.no_augment)
    val_ds = PlateDataset(args.val_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
    )

    print(f"Dispositivo: {device} | Train: {len(train_ds)} placas | Val: {len(val_ds)} placas")
    print(f"Épocas: {args.epochs} | Batch: {args.batch_size} | LR inicial: {args.lr} | Aumentos: {not args.no_augment}")
    print("-" * 82)

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr_actual = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, ejemplos = validate(model, val_loader, criterion, device)
        scheduler.step()

        duracion = time.time() - t0
        val_cer = val_metrics["cer"]
        val_acc = val_metrics["plate_acc"]

        marca = ""
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), out_path)
            marca = " [MEJOR GUARDADO]"

        print(
            f"Época {epoch:3d}/{args.epochs} ({duracion:4.1f}s) | LR: {lr_actual:.6f} | "
            f"TrainLoss: {train_loss:6.3f} | ValLoss: {val_loss:6.3f} | "
            f"ValAcc: {val_acc*100:5.1f}% | ValCER: {val_cer*100:5.1f}%{marca}"
        )

        # Mostrar 4 placas de validación (nunca vistas al entrenar) cada 5 épocas o en la primera/última
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            muestra = [
                f"{real}->{pred or '(vacío)'}({conf:.2f})"
                for real, pred, conf in ejemplos
            ]
            print("   Val ejemplos:", " | ".join(muestra))

    print("-" * 82)
    print(f"Entrenamiento finalizado. Mejor ValCER: {best_cer*100:.2f}% | Pesos guardados en: {out_path}")


if __name__ == "__main__":
    main()