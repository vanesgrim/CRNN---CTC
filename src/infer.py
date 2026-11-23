from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import requests
import torch

from src.dataset import NUM_CLASSES, preprocess_plate
from src.model import CRNN
from src.utils import greedy_decode, postprocess_plate

# Patrones regex predefinidos para el Paso 8 (postprocesamiento)
REGEX_PRESETS: dict[str, str | None] = {
    "es_full": r"^[0-9]{4}[A-Z]{3}$",          # Matrícula española completa (ej. 1234BCD)
    "es_uc3m": r"^[0-9]{1,4}[A-Z]{1,3}$",      # Formato UC3M-LP (con caracteres difuminados omitidos)
    "mx": r"^[A-Z]{3}[0-9]{3,4}[A-Z]?$",       # Formato común en México (ej. ABC1234 / ABC123A)
    "any": None,                               # Solo limpia caracteres no alfanuméricos
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_regex(regex_arg: str | None) -> str | None:
    """Permite usar un preset ('es_full', 'es_uc3m', 'mx', 'any') o una regex personalizada."""
    if regex_arg is None:
        return None
    return REGEX_PRESETS.get(regex_arg, regex_arg)


def decode_bytes_to_bgr(raw_bytes: bytes | bytearray | memoryview) -> np.ndarray:
    """Decodifica bytes de imagen (JPG/PNG) directo en RAM a matriz BGR sin escribir a disco."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudieron decodificar los bytes como una imagen válida.")
    return img_bgr


class PlateRecognizer:
    """Motor de inferencia unificado sobre la CRNN entrenada."""

    def __init__(
        self,
        weights_path: str | Path = "checkpoints/crnn_best.pt",
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"No se encontró el checkpoint: {self.weights_path}")

        self.model = CRNN(num_classes=NUM_CLASSES).to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict_from_bgr(
        self,
        img_bgr: np.ndarray,
        pattern: str | None = None,
        two_lines: bool | None = None,
    ) -> dict[str, Any]:
        """Núcleo único: recibe una placa BGR en RAM y devuelve texto, confianza, validez y latencia."""
        t0 = time.perf_counter()

        h, w = img_bgr.shape[:2]
        is_two_lines = (w / h) < 1.6 if two_lines is None else two_lines

        tensor = preprocess_plate(img_bgr, two_lines=is_two_lines)  # (1, 32, 128)
        batch = tensor.unsqueeze(0).to(self.device)                 # (1, 1, 32, 128)

        logits = self.model(batch)                                  # (1, 32, 37)
        raw_text, confidence = greedy_decode(logits.cpu())[0]
        cleaned_text, is_valid = postprocess_plate(raw_text, pattern=pattern)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "plate": cleaned_text,
            "raw": raw_text,
            "confidence": round(confidence, 4),
            "valid_format": is_valid,
            "two_lines": is_two_lines,
            "latency_ms": round(latency_ms, 2),
        }

    # --- Adaptador 1: Bytes en RAM ---
    def predict_from_bytes(
        self,
        raw_bytes: bytes | bytearray | memoryview,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        img_bgr = decode_bytes_to_bgr(raw_bytes)
        result = self.predict_from_bgr(img_bgr, pattern=pattern)
        result["source"] = "memory_bytes"
        return result

    # --- Adaptador 2: Disco local ---
    def predict_from_disk(
        self,
        image_path: str | Path,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        path = Path(image_path)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo abrir la imagen en disco: {path}")
        result = self.predict_from_bgr(img_bgr, pattern=pattern)
        result["source"] = f"disk:{path.name}"
        return result

    # --- Adaptador 3: Memoria compartida con Kax ---
    def predict_from_kax_shm(
        self,
        kax_client: Any,
        shm_name: str,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        """Lee los bytes crudos desde memoria compartida gestionada por Kax sin tocar el disco."""
        raw_bytes = kax_client.getRes(shm_name)
        if raw_bytes is None:
            raise KeyError(f"No se encontró el segmento en memoria compartida de Kax: '{shm_name}'")
        result = self.predict_from_bytes(raw_bytes, pattern=pattern)
        result["source"] = f"kax_shm:{shm_name}"
        return result

    # --- Adaptador 4A: Cloud vía HTTP/HTTPS directo ---
    def predict_from_url(
        self,
        url: str,
        pattern: str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Descarga la imagen por red directo a RAM (sin archivos temporales)."""
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        result = self.predict_from_bytes(resp.content, pattern=pattern)
        result["source"] = f"cloud_http:{url}"
        return result

    # --- Adaptador 4B: Cloud vía Kax Storage Client ---
    def predict_from_kax_cloud(
        self,
        kax_client: Any,
        hash_id: str,
        pattern: str | None = None,
    ) -> dict[str, Any]:
        """Descarga la imagen desde el servicio cloud de Kax usando su hash_id."""
        payload = kax_client.download_from_cloud(hash_id)

        if isinstance(payload, (bytes, bytearray, memoryview)):
            result = self.predict_from_bytes(payload, pattern=pattern)
        elif isinstance(payload, (str, Path)) and Path(payload).exists():
            result = self.predict_from_disk(payload, pattern=pattern)
        else:
            # Si download_from_cloud depositó los datos en SHM bajo hash_id, los leemos de ahí
            shm_bytes = kax_client.getRes(hash_id)
            if shm_bytes is None:
                raise RuntimeError(
                    f"download_from_cloud('{hash_id}') no devolvió bytes ni cargó el segmento en SHM."
                )
            result = self.predict_from_bytes(shm_bytes, pattern=pattern)

        result["source"] = f"kax_cloud:{hash_id}"
        return result


def main():
    parser = argparse.ArgumentParser(description="Inferencia ALPR desde disco, kax SHM o cloud")
    parser.add_argument(
        "--source",
        choices=["disk", "kax_shm", "url", "kax_cloud"],
        required=True,
        help="Origen de la imagen",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta local (disk), nombre/hash SHM (kax_shm), URL (url) o hash_id (kax_cloud)",
    )
    parser.add_argument("--weights", default="checkpoints/crnn_best.pt", help="Ruta al checkpoint .pt")
    parser.add_argument(
        "--regex",
        default="es_uc3m",
        help="Preset (es_full, es_uc3m, mx, any) o expresión regular personalizada",
    )
    parser.add_argument("--registry-url", default="http://localhost:5060/api")
    parser.add_argument("--storage-url", default="http://localhost:5070/api")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    pattern = resolve_regex(args.regex)
    recognizer = PlateRecognizer(weights_path=args.weights)

    if args.source == "disk":
        res = recognizer.predict_from_disk(args.input, pattern=pattern)
    elif args.source == "url":
        res = recognizer.predict_from_url(args.input, pattern=pattern)
    else:
        from kax import Kax

        kax_client = Kax(
            registry_url=args.registry_url,
            storage_url=args.storage_url,
            broker=args.broker,
            port=args.port,
        )
        if args.source == "kax_shm":
            res = recognizer.predict_from_kax_shm(kax_client, args.input, pattern=pattern)
        else:
            res = recognizer.predict_from_kax_cloud(kax_client, args.input, pattern=pattern)

    print(res)


if __name__ == "__main__":
    main()