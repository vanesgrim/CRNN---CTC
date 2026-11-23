from __future__ import annotations

import argparse
import errno
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kax import Kax
from src.infer import PlateRecognizer, resolve_regex


def extract_hash(metadata: Any) -> str | None:
    """Extrae el identificador hash desde la metadata que devuelve Kax."""
    if isinstance(metadata, dict):
        resources = metadata.get("resources")
        if isinstance(resources, list) and resources:
            first = resources[0]
            if isinstance(first, dict) and "hash" in first:
                return str(first["hash"])
        if "hash" in metadata:
            return str(metadata["hash"])
    elif isinstance(metadata, list) and metadata:
        first = metadata[0]
        if isinstance(first, dict) and "hash" in first:
            return str(first["hash"])
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prueba de extremo a extremo: subir placa -> descargar de Kax Cloud -> inferir CRNN -> limpiar"
    )
    parser.add_argument(
        "--registry-url",
        default=os.getenv("KAX_REGISTRY_URL"),
        help="URL del servicio de registro (ej. http://IP:5060/api)",
    )
    parser.add_argument(
        "--storage-url",
        default=os.getenv("KAX_STORAGE_URL"),
        help="URL del servicio de almacenamiento (ej. http://IP:5070/api)",
    )
    parser.add_argument(
        "--broker",
        default=os.getenv("KAX_BROKER"),
        help="Host o IP del broker MQTT",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KAX_PORT", "1883")),
        help="Puerto del broker MQTT (por defecto 1883)",
    )
    parser.add_argument(
        "--image",
        default="data/raw/imgs/val/00014_AN-07_6_YR.jpg",
        help="Placa de prueba para subir y leer",
    )
    parser.add_argument("--weights", default="checkpoints/crnn_best.pt", help="Ruta al checkpoint .pt")
    parser.add_argument("--regex", default="es_uc3m", help="Preset o expresión regular")
    parser.add_argument(
        "--upload-mode",
        choices=["disk", "memory"],
        default="disk",
        help="disk = uploadToCloudFromDisk | memory = loadAndUploadToCloud",
    )
    parser.add_argument(
        "--keep-cloud",
        action="store_true",
        help="No borrar el segmento en la nube al terminar (por defecto SIEMPRE se borra)",
    )
    args = parser.parse_args()

    if not args.registry_url or not args.storage_url or not args.broker:
        raise SystemExit(
            "Faltan datos de conexión. Pasa --registry-url, --storage-url y --broker "
            "(o define KAX_REGISTRY_URL, KAX_STORAGE_URL y KAX_BROKER)."
        )

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen de prueba: {img_path}")

    pattern = resolve_regex(args.regex)
    recognizer = PlateRecognizer(weights_path=args.weights)

    print("=== 1. Referencia local (DISCO) ===")
    ref = recognizer.predict_from_disk(img_path, pattern=pattern)
    print(f"  Imagen: {img_path} ({img_path.stat().st_size:,} bytes)")
    print(f"  Disco:  plate='{ref['plate']}' | conf={ref['confidence']} | regex_ok={ref['valid_format']}")

    print("\n=== 2. Conexión a Kax Cloud ===")
    print(f"  registry_url: {args.registry_url}")
    print(f"  storage_url:  {args.storage_url}")
    print(f"  broker:       {args.broker}:{args.port}")

    client = Kax(
        registry_url=args.registry_url,
        storage_url=args.storage_url,
        broker=args.broker,
        port=args.port,
        workers=1,
    )

    hash_id: str | None = None
    uploaded = False

    try:
        print(f"\n=== 3. Subiendo 1 placa de prueba ({args.upload_mode}) ===")
        if args.upload_mode == "disk":
            metadata = client.uploadToCloudFromDisk(str(img_path))
        else:
            metadata = client.loadAndUploadToCloud(str(img_path))

        hash_id = extract_hash(metadata)
        if not hash_id:
            raise RuntimeError(f"No se pudo extraer 'hash' de la respuesta de Kax: {metadata}")

        uploaded = True
        print(f"  Subida exitosa. hash_id = {hash_id}")

        # Si subimos desde memoria, liberamos primero la copia local para garantizar
        # que la lectura siguiente venga realmente de la descarga desde la nube.
        if args.upload_mode == "memory":
            try:
                client.removeShm(hash_id)
            except Exception:
                pass

        print("\n=== 4. Descargando desde Kax Cloud e infiriendo en RAM ===")
        cloud_res = recognizer.predict_from_kax_cloud(client, hash_id, pattern=pattern)
        coincide = cloud_res["plate"] == ref["plate"]

        print(f"  Resultado Cloud: {cloud_res}")
        print(f"  ¿Coincide 100% con disco ('{ref['plate']}')?: {coincide}")

    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            print(f"\n[Límite de macOS detectado] {exc}")
            print(
                "  Causa: Kax usa el hash SHA-256 (64 caracteres) como nombre de segmento POSIX,\n"
                "  pero macOS solo permite hasta 31 caracteres. En Ubuntu (límite 255 caracteres)\n"
                "  este mismo comando correrá completo sin modificar el código."
            )
        else:
            raise

    finally:
        if hash_id is not None:
            print("\n=== 5. Limpieza automática (local y nube) ===")
            try:
                msg_local = client.removeShm(hash_id)
                print(f"  removeShm (local):          {msg_local}")
            except Exception as exc:
                print(f"  removeShm (local):          sin segmento local activo ({exc})")

            if uploaded and not args.keep_cloud:
                try:
                    msg_cloud = client.removeShmFromCloud(hash_id)
                    print(f"  removeShmFromCloud (nube):  {msg_cloud}")
                except Exception as exc:
                    print(f"  [Aviso] No se pudo borrar en nube ({hash_id}): {exc}")


if __name__ == "__main__":
    main()





