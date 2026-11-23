import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kax import Kax, mem as kax_mem
from src.dataset import VALID_EXTS, parse_label_from_filename
from src.infer import PlateRecognizer, resolve_regex
from src.utils import postprocess_plate


def probar_disco(recognizer: PlateRecognizer, val_images: list[Path]) -> Path:
    print("=== PRUEBA 1: Inferencia desde DISCO (placas de val nunca vistas) ===")
    pattern = resolve_regex("es_uc3m")

    for img_path in val_images[:6]:
        real = parse_label_from_filename(img_path.name)
        res = recognizer.predict_from_disk(img_path, pattern=pattern)
        estado = "EXACTA" if res["plate"] == real else "DIFIERE"
        print(
            f"  [{estado:7s}] {img_path.name:24s} | Real={real:6s} -> Pred={res['plate']:6s} "
            f"| Conf={res['confidence']:.3f} | RegexOK={str(res['valid_format']):5s} | {res['latency_ms']:5.1f} ms"
        )
    return val_images[1]  # Elegimos una placa bien leída para comparar las 3 fuentes


def probar_kax_memoria(recognizer: PlateRecognizer, img_path: Path, ref_plate: str):
    print("\n=== PRUEBA 2: Inferencia desde MEMORIA COMPARTIDA (kax) ===")
    kax_client = Kax(workers=1)
    pattern = resolve_regex("es_uc3m")
    shm_name = None

    try:
        # Intento principal: flujo nativo completo de Kax (load_res -> getRes -> removeShm)
        metadata = kax_client.load_res(str(img_path), load_to_memory=True)
        shm_name = metadata["resources"][0]["hash"]
        res = recognizer.predict_from_kax_shm(kax_client, shm_name, pattern=pattern)
        modo = "kax.load_res + kax.getRes"
    except OSError as exc:
        # En macOS, POSIX SharedMemory rechaza nombres >31 chars (ej. SHA-256); en Ubuntu funciona directo.
        print(f"  [Aviso macOS] kax.load_res reportó límite del SO ({exc}).")
        print("  Probando memoria compartida de kax con nombre corto (makeShm + getRes)...")
        raw_bytes = img_path.read_bytes()
        shm_name = "placa_test_shm"
        kax_client.makeShm(shm_name, len(raw_bytes))
        kax_mem.putDataToLocalSharedMemory(shm_name, raw_bytes, 0, len(raw_bytes))
        res = recognizer.predict_from_kax_shm(kax_client, shm_name, pattern=pattern)
        modo = "kax.makeShm + kax.getRes"
    finally:
        if shm_name is not None:
            try:
                kax_client.removeShm(shm_name)
            except Exception:
                pass

    iguales = res["plate"] == ref_plate
    print(f"  Modo usado: {modo}")
    print(f"  Resultado:  {res}")
    print(f"  ¿Coincide 100% con disco ('{ref_plate}')?: {iguales}")


def probar_cloud_http(recognizer: PlateRecognizer, img_path: Path, ref_plate: str):
    print("\n=== PRUEBA 3: Inferencia desde CLOUD (HTTP directo a RAM) ===")
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(img_path.parent),
    )

    # Servidor silencioso en un puerto libre temporal solo para probar HTTP -> bytes -> CRNN
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    handler = functools.partial(QuietHandler, directory=str(img_path.parent))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}/{img_path.name}"
            res = recognizer.predict_from_url(url, pattern=resolve_regex("es_uc3m"))
        finally:
            httpd.shutdown()

    iguales = res["plate"] == ref_plate
    print(f"  URL servida: {url}")
    print(f"  Resultado:   {res}")
    print(f"  ¿Coincide 100% con disco ('{ref_plate}')?: {iguales}")


def probar_regex():
    print("\n=== PRUEBA 4: Validación por Expresiones Regulares (Paso 8) ===")
    casos = [
        ("1234BCD", "es_full", "Española completa (4 dígitos + 3 letras)"),
        ("076YR", "es_full", "Parcial UC3M-LP contra formato español completo"),
        ("076YR", "es_uc3m", "Parcial UC3M-LP contra formato UC3M-LP"),
        ("XAA123A", "mx", "Formato común México"),
        ("??99--", "es_uc3m", "Lectura ruidosa sin letras"),
    ]
    for texto, preset, desc in casos:
        limpio, valido = postprocess_plate(texto, pattern=resolve_regex(preset))
        print(f"  '{texto:8s}' -> limpio='{limpio:7s}' | preset={preset:7s} | válido={str(valido):5s} | {desc}")


if __name__ == "__main__":
    val_dir = Path("data/raw/imgs/val")
    val_images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in VALID_EXTS)

    recognizer = PlateRecognizer("checkpoints/crnn_best.pt")
    print(f"Modelo cargado desde checkpoints/crnn_best.pt en dispositivo: {recognizer.device}\n")

    img_muestra = probar_disco(recognizer, val_images)
    ref_plate = recognizer.predict_from_disk(img_muestra, pattern=resolve_regex("es_uc3m"))["plate"]

    probar_kax_memoria(recognizer, img_muestra, ref_plate)
    probar_cloud_http(recognizer, img_muestra, ref_plate)
    probar_regex()