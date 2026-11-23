import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import NUM_CLASSES, PlateDataset, ctc_collate_fn
from src.model import CRNN
from src.utils import evaluate_predictions, greedy_decode


def elegir_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    torch.manual_seed(42)
    device = elegir_dispositivo()
    print(f"Dispositivo activo: {device}")

    # 1. Tomar únicamente 1 lote fijo de 16 placas reales (sin aumentos)
    ds = PlateDataset("data/raw/imgs/train", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ctc_collate_fn)
    images, targets, target_lengths, labels = next(iter(loader))

    images = images.to(device)
    if device.type != "mps":
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

    print(f"Lote fijo de prueba: {len(labels)} placas -> primeras 4: {labels[:4]}")
    print("-" * 72)

    # 2. Instanciar modelo, función de pérdida CTC y optimizador Adam (lr = 1e-3)
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Todas las imágenes producen T = 32 columnas a la salida de la CNN
    input_lengths = torch.full(
        size=(images.size(0),),
        fill_value=32,
        dtype=torch.long,
        device="cpu" if device.type == "mps" else device,
    )

    pasos_totales = 200
    inicio = time.time()

    for paso in range(1, pasos_totales + 1):
        model.train()

        # --- Los 4 pasos del ciclo de entrenamiento ---
        optimizer.zero_grad()

        logits = model(images)                                  # (16, 32, 37)
        log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)  # (32, 16, 37) para CTCLoss

        if device.type == "mps":
            loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)
        else:
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Mostrar progreso en el paso 1 y cada 20 pasos
        if paso == 1 or paso % 20 == 0 or paso == pasos_totales:
            model.eval()
            with torch.no_grad():
                eval_logits = model(images)
            decodificadas = greedy_decode(eval_logits.cpu())
            preds = [texto for texto, _ in decodificadas]
            metricas = evaluate_predictions(preds, labels)

            print(
                f"Paso {paso:3d}/{pasos_totales} | "
                f"Loss: {loss.item():6.3f} | "
                f"Acc Placa: {metricas['plate_acc']*100:5.1f}% | "
                f"CER: {metricas['cer']*100:5.1f}%"
            )
            # Mostrar 4 ejemplos para ver evolucionar la lectura con tus ojos
            ejemplos = [
                f"{real}->{pred or '(vacío)'}({conf:.2f})"
                for real, (pred, conf) in zip(labels[:4], decodificadas[:4])
            ]
            print("   Ejemplos:", " | ".join(ejemplos))

            # Si ya leyó las 16 placas perfectas con pérdida mínima, terminamos antes
            if metricas["plate_acc"] == 1.0 and loss.item() < 0.05:
                print("-" * 72)
                print(f"¡Prueba superada en {paso} pasos ({time.time() - inicio:.1f} s)!")
                break


if __name__ == "__main__":
    main()