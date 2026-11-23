import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from src.dataset import CHAR_TO_IDX, NUM_CLASSES, PlateDataset, ctc_collate_fn
from src.model import CRNN
from src.utils import ctc_collapse, evaluate_predictions, greedy_decode, levenshtein_distance


def cadena_a_indices_ctc(cadena: str) -> list[int]:
    """Convierte una cadena de prueba con '-' como blanco en índices CTC (0..36)."""
    return [0 if c == "-" else CHAR_TO_IDX[c] for c in cadena]


print("=== PRUEBA 1: Colapso CTC B(pi) con secuencias inventadas ===")
casos_ctc = [
    ("---00-777--4-C-VV---", "074CV", "Repetidos normales + blancos"),
    ("44-4-44--P--CC", "444PC", "El blanco '-' permite repetir el '4' tres veces"),
    ("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYN", "YN", "Tu salida real del Paso 5 sin entrenar"),
    ("--------------------------------", "", "Solo blancos -> cadena vacía"),
]

for cruda, esperado, descripcion in casos_ctc:
    indices = cadena_a_indices_ctc(cruda)
    obtenido = ctc_collapse(indices)
    ok = "OK" if obtenido == esperado else "ERROR"
    print(f"  [{ok}] {cruda:34s} -> '{obtenido:6s}' (esperado: '{esperado}') | {descripcion}")

print("\n=== PRUEBA 2: Distancia de Levenshtein, Exactitud por placa y CER ===")
reales = ["074CV", "074CV", "074CV", "074CV"]
preds  = ["074CV", "074CW", "07CV",  "074CVX"]
notas  = ["Exacta (0 errores)", "1 sustitución (W vs V)", "1 borrado (faltó 4)", "1 inserción (sobró X)"]

for p, r, nota in zip(preds, reales, notas):
    dist = levenshtein_distance(p, r)
    print(f"  Real='{r}' vs Pred='{p:6s}' -> distancia={dist} | {nota}")

metricas = evaluate_predictions(preds, reales)
print(f"  -> Exactitud por placa: {metricas['plate_acc']*100:.1f}% (esperado: 25.0%, 1 de 4)")
print(f"  -> CER:                 {metricas['cer']*100:.1f}% (esperado: 15.0%, 3 ediciones / 20 caracteres)")

print("\n=== PRUEBA 3: Conectar greedy_decode directo a la CRNN (sin entrenar) ===")
ds = PlateDataset("data/raw/imgs/train", augment=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ctc_collate_fn)
images, _, _, labels = next(iter(loader))

model = CRNN(num_classes=NUM_CLASSES)
model.eval()
with torch.no_grad():
    logits = model(images)

decodificadas = greedy_decode(logits)
preds_crnn = [texto for texto, _ in decodificadas]

for real, (pred, conf) in zip(labels, decodificadas):
    print(f"  Real='{real}' | Pred CRNN='{pred:6s}' | Confianza={conf:.6f}")

m_crnn = evaluate_predictions(preds_crnn, labels)
print(f"  -> Métricas sin entrenar: Acc Placa={m_crnn['plate_acc']*100:.1f}% | CER={m_crnn['cer']*100:.1f}%")