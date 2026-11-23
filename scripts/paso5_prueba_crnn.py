import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import IDX_TO_CHAR, NUM_CLASSES, PlateDataset, ctc_collate_fn
from src.model import CRNN

# 1. Tomar 1 lote real de 4 placas
ds = PlateDataset("data/raw/imgs/train", augment=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ctc_collate_fn)
images, _, _, labels = next(iter(loader))

print("Placas reales en este lote:", labels)
print("-" * 65)

# 2. Instanciar la red CRNN completa
model = CRNN(num_classes=NUM_CLASSES)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros totales en la CRNN (CNN + BiLSTM + Linear): {total_params:,}")
print("-" * 65)

# 3. Pasar las 4 placas por toda la red mostrando formas en cada etapa
with torch.no_grad():
    logits = model(images, mostrar_formas=True)

# 4. Convertir logits crudos -> probabilidades (softmax sobre las 37 clases en dim=2)
probs = F.softmax(logits, dim=2)

# Revisar la columna t=0 de la primera placa
suma_probs_t0 = probs[0, 0].sum().item()
top3_vals, top3_idxs = torch.topk(probs[0, 0], k=3)

print("-" * 65)
print(f"Suma de las 37 probabilidades en t=0 (debe ser 1.0): {suma_probs_t0:.6f}")
print("Top 3 clases en t=0 de la placa 1 (red sin entrenar):")
for val, idx in zip(top3_vals.tolist(), top3_idxs.tolist()):
    simbolo = "ε (blanco)" if idx == 0 else f"'{IDX_TO_CHAR[idx]}'"
    print(f"  índice {idx:2d} -> {simbolo:11s} | prob = {val:.4f} ({val*100:.2f}%)")

# 5. Ver los 32 caracteres crudos (argmax) que eligió la red para la placa 1
indices_crudos = probs[0].argmax(dim=1).tolist()
simbolos_crudos = ["-" if i == 0 else IDX_TO_CHAR[i] for i in indices_crudos]
print("-" * 65)
print(f"Etiqueta real de placa 1:          {labels[0]}")
print(f"32 pasos crudos (sin entrenar):    {''.join(simbolos_crudos)}")
print("Nota: '-' representa el símbolo blanco ε; como aún no entrenamos, verás letras casi uniformes (~2.7% cada una).")