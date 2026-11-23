from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Alfabeto del documento: 36 caracteres (1..36) + símbolo blanco CTC en el índice 0 -> C = 37
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_from_filename(filename: str) -> str:
    """
    Extrae la placa real después del guion '-' y elimina '_' (caracteres difuminados).
    Ejemplo: '00001_AN-074_C_V.jpg' -> '074CV'
    """
    stem = Path(filename).stem
    raw_label = stem.split("-", 1)[1] if "-" in stem else stem
    return "".join(c for c in raw_label.upper() if c in CHAR_TO_IDX)


def encode_label(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text.upper() if c in CHAR_TO_IDX]


# --- Funciones individuales de aumento de datos (Paso 2 del documento) ---

def aug_brightness_contrast(img: np.ndarray, alpha: float | None = None, beta: float | None = None) -> np.ndarray:
    a = alpha if alpha is not None else float(np.random.uniform(0.8, 1.25))
    b = beta if beta is not None else float(np.random.uniform(-20, 20))
    return cv2.convertScaleAbs(img, alpha=a, beta=b)


def aug_gaussian_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def aug_motion_blur(img: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """Desenfoque direccional horizontal para simular movimiento del vehículo."""
    k = kernel_size if kernel_size is not None else int(np.random.choice([5, 7]))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def aug_gaussian_noise(img: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Ruido aleatorio por píxel para simular cámaras de baja luz."""
    s = sigma if sigma is not None else float(np.random.uniform(8.0, 20.0))
    noise = np.random.normal(0.0, s, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def aug_rotation(img: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Rotación pequeña dentro del rango [-5°, +5°]."""
    h, w = img.shape[:2]
    deg = angle if angle is not None else float(np.random.uniform(-5.0, 5.0))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def aug_random_erasing(img: np.ndarray) -> np.ndarray:
    """Tapa un rectángulo pequeño al azar para simular suciedad, tornillos u oclusiones."""
    out = img.copy()
    h, w = out.shape[:2]
    erase_h = max(2, int(h * np.random.uniform(0.18, 0.35)))
    erase_w = max(2, int(w * np.random.uniform(0.08, 0.16)))
    y = int(np.random.randint(0, max(1, h - erase_h + 1)))
    x = int(np.random.randint(0, max(1, w - erase_w + 1)))
    fill_val = int(np.random.randint(0, 256))
    out[y : y + erase_h, x : x + erase_w] = fill_val
    return out


def apply_augmentations(img: np.ndarray) -> np.ndarray:
    """Aplica aleatoriamente una combinación suave de los 6 aumentos durante entrenamiento."""
    if np.random.rand() < 0.4:
        img = aug_brightness_contrast(img)
    if np.random.rand() < 0.2:
        img = aug_gaussian_blur(img)
    if np.random.rand() < 0.2:
        img = aug_motion_blur(img)
    if np.random.rand() < 0.2:
        img = aug_gaussian_noise(img)
    if np.random.rand() < 0.2:
        img = aug_rotation(img)
    if np.random.rand() < 0.15:
        img = aug_random_erasing(img)
    return img


def preprocess_plate(img_bgr: np.ndarray, two_lines: bool = False) -> torch.Tensor:
    """
    Convierte un recorte BGR al tensor (1, 32, 128) en escala de grises,
    conservando proporción con relleno negro a la derecha y normalizado en [-1, 1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if two_lines:
        h = gray.shape[0]
        top = gray[: h // 2, :]
        bottom = gray[h // 2 :, :]
        h_min = min(top.shape[0], bottom.shape[0])
        gray = np.hstack([top[:h_min, :], bottom[:h_min, :]])

    h, w = gray.shape
    new_w = min(128, max(1, int(round(w * (32.0 / h)))))
    resized = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((32, 128), dtype=np.uint8)
    canvas[:, :new_w] = resized

    img_norm = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(img_norm).unsqueeze(0)  # Forma final: (1, 32, 128)


class PlateDataset(Dataset):
    def __init__(self, folder: str | Path, augment: bool = False):
        self.folder = Path(folder)
        self.augment = augment
        self.samples: list[tuple[Path, str]] = []

        for path in sorted(self.folder.iterdir()):
            if path.suffix.lower() not in VALID_EXTS:
                continue
            label = parse_label_from_filename(path.name)
            if label:
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

        if self.augment:
            img = apply_augmentations(img)

        h, w = img.shape[:2]
        two_lines = (w / h) < 1.6

        tensor = preprocess_plate(img, two_lines=two_lines)
        target = torch.tensor(encode_label(label), dtype=torch.long)
        return tensor, target, label


def ctc_collate_fn(batch):
    images, targets, raw_labels = zip(*batch)
    images = torch.stack(images, dim=0)                              # (N, 1, 32, 128)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)                       # Vector 1D continuo
    return images, targets_concat, target_lengths, list(raw_labels)