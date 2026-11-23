from pathlib import Path
import cv2
import numpy as np


#busca la carpeta donde se encuentran las imagenes de entrenamiento
RUTAS = [Path("data/raw/imgs/train"), Path("raw/imgs/train")]
CARPETA_TRAIN = next((p for p in RUTAS if p.exists()), None)

if CARPETA_TRAIN is None:
    raise FileNotFoundError("No se encontró la carpeta de imágenes de entrenamiento.")  

CARPETA_DEBUG = Path("data/processed/debug")
CARPETA_DEBUG.mkdir(parents=True, exist_ok=True)

#tomar la primera imagen valida que encuentre en train
EXIST = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
imagenes = sorted([f for f in CARPETA_TRAIN.rglob("*") if f.suffix.lower() in EXIST])

if not imagenes:
    raise FileNotFoundError("No se encontraron imágenes válidas en la carpeta de entrenamiento.")

img_path = imagenes[0]
img_bgr = cv2.imread(str(img_path))

if img_bgr is None:
    raise ValueError(f"No se pudo leer la imagen: {img_path}")

print(f"Imagen cargada correctamente: {img_path}")

print(f"[1. Original bgr] forma: {img_bgr.shape}, dtype: {img_bgr.dtype}")

cv2.imwrite(str(CARPETA_DEBUG / "1_original_bgr.png"), img_bgr)

# paso 2 conversion a escala de grises (pasa de 3 canales a 1 canal)

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"[2. Escala de grises] forma: {img_gray.shape}, dtype: {img_gray.dtype}")
cv2.imwrite(str(CARPETA_DEBUG / "2_escala_grises.png"), img_gray)

# 3 redimensionar la imagen a un tamaño fijo (por ejemplo, 32x128)
h, w = img_gray.shape
nuevo_w = min(128, max(1, int(round(w * 32 / h))))  # Mantener la relación de aspecto

resize_img = cv2.resize(img_gray, (nuevo_w, 32), interpolation=cv2.INTER_AREA)
print(f"[3. Redimensionada] forma: {resize_img.shape}, dtype: {resize_img.dtype}")
cv2.imwrite(str(CARPETA_DEBUG / "3_redimensionada.png"), resize_img)

# 4 relleno de la imagen para que tenga un tamaño fijo de 32x128
canvas = np.zeros((32, 128), dtype=np.uint8)
canvas[:, :nuevo_w] = resize_img
print(f"[4. Rellenada] forma: {canvas.shape}, dtype: {canvas.dtype}")
cv2.imwrite(str(CARPETA_DEBUG / "4_rellenada.png"), canvas)

# 5 normalización de la imagen, (x - mu) / sigma, donde mu es la media y sigma es la desviación estándar de los píxeles de la imagen
img_norm = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5  # Normalización a [-1, 1]
print(f"[5. Normalizada]   forma={img_norm.shape} | tipo={img_norm.dtype} | min={img_norm.min():.3f} max={img_norm.max():.3f} | media={img_norm.mean():.3f}")
print("\nListo. Abre las 4 imágenes guardadas en:", CARPETA_DEBUG)