import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.dataset import (
    VALID_EXTS,
    apply_augmentations,
    aug_brightness_contrast,
    aug_gaussian_blur,
    aug_gaussian_noise,
    aug_motion_blur,
    aug_random_erasing,
    aug_rotation,
)

CARPETA_TRAIN = Path("data/raw/imgs/train")
CARPETA_SALIDA = Path("data/processed/debug/aumentos")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Elegir una placa con buena iluminación para notar bien los efectos visuales
imagenes = sorted(p for p in CARPETA_TRAIN.iterdir() if p.suffix.lower() in VALID_EXTS)
img_path = imagenes[0]
img = cv2.imread(str(img_path))

for p in imagenes[:50]:
    candidata = cv2.imread(str(p))
    if candidata is not None and candidata.mean() > 90:
        img_path = p
        img = candidata
        break

print(f"Placa elegida para prueba: {img_path.name} | forma={img.shape} | brillo promedio={img.mean():.1f}")

# Generar cada transformación por separado (con semilla fija para reproducibilidad)
np.random.seed(42)
versiones = [
    ("1_original", "1. Original", img),
    ("2_brillo_contraste", "2. Brillo y contraste", aug_brightness_contrast(img, alpha=1.25, beta=20)),
    ("3_desenfoque_gaussiano", "3. Desenfoque gaussiano", aug_gaussian_blur(img, kernel_size=5)),
    ("4_desenfoque_movimiento", "4. Desenfoque de movimiento (k=9)", aug_motion_blur(img, kernel_size=9)),
    ("5_ruido_gaussiano", "5. Ruido gaussiano (sigma=20)", aug_gaussian_noise(img, sigma=20.0)),
    ("6_rotacion", "6. Rotacion (+5 grados)", aug_rotation(img, angle=5.0)),
    ("7_borrado_aleatorio", "7. Borrado aleatorio (random erasing)", aug_random_erasing(img)),
    ("8_combinado", "8. Combinado aleatorio (entrenamiento)", apply_augmentations(img.copy())),
]

paneles = []
ancho_panel = 380

for nombre_archivo, titulo, resultado in versiones:
    # Guardar imagen individual
    cv2.imwrite(str(CARPETA_SALIDA / f"{nombre_archivo}.jpg"), resultado)

    # Preparar panel etiquetado para el resumen visual
    h, w = resultado.shape[:2]
    nuevo_h = int(round(h * (ancho_panel / w)))
    redim = cv2.resize(resultado, (ancho_panel, nuevo_h), interpolation=cv2.INTER_AREA)

    barra = np.zeros((28, ancho_panel, 3), dtype=np.uint8)
    cv2.putText(barra, titulo, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    paneles.append(np.vstack([barra, redim]))

resumen = np.vstack(paneles)
ruta_resumen = CARPETA_SALIDA / "00_resumen_aumentos.jpg"
cv2.imwrite(str(ruta_resumen), resumen)

print(f"Listo. Se guardaron {len(versiones)} imágenes individuales y el resumen en:")
print(" ", ruta_resumen)