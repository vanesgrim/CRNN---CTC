import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from src.dataset import PlateDataset, ctc_collate_fn
from src.model import CNNExtractor, mapa_a_secuencia

# 1. Tomar 1 lote real de 4 placas desde train
ds = PlateDataset("data/raw/imgs/train", augment=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ctc_collate_fn)
images, _, _, labels = next(iter(loader))

print("Placas reales en este lote:", labels)
print("-" * 65)

# 2. Instanciar la CNN (en modo eval porque solo estamos haciendo una prueba sin entrenar)
cnn = CNNExtractor()
cnn.eval()

num_params = sum(p.numel() for p in cnn.parameters())
print(f"Parámetros entrenables en la CNN: {num_params:,}")
print("-" * 65)

# 3. Pasar las 4 placas por los 5 bloques imprimiendo las dimensiones en cada etapa
with torch.no_grad():
    mapa = cnn(images, mostrar_formas=True)
    secuencia = mapa_a_secuencia(mapa)

print("-" * 65)
print(f"Secuencia final (N, T, F): {tuple(secuencia.shape)}")
print("¿Hay algún valor inválido (NaN)?:", torch.isnan(secuencia).any().item())

# Ver los primeros 5 de los 512 números que describen la primera columna (t=1) de la placa 1
primer_vector = secuencia[0, 0, :5].tolist()
print("Primeros 5 números del vector f_1 (placa 1):", [round(v, 4) for v in primer_vector])