from __future__ import annotations

import argparse
import errno
import gc
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from kax import Kax

from src.infer import PlateRecognizer, resolve_regex


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def format_size_label(target_kb: int) -> str:
    if target_kb >= 1048576 and target_kb % 1048576 == 0:
        return f"{target_kb // 1048576}GB"
    if target_kb >= 1024:
        return f"{target_kb // 1024}MB"
    return f"{target_kb}KB"


def summarize_metrics(
    io_ms: list[float],
    total_ms: list[float],
    payload_bytes: int,
    integrity_ok: int,
    total_runs: int,
) -> dict[str, Any]:
    io_arr = np.array(io_ms, dtype=np.float64)
    tot_arr = np.array(total_ms, dtype=np.float64)
    mean_io = float(io_arr.mean())
    mean_tot = float(tot_arr.mean())
    size_mb = payload_bytes / (1024.0 * 1024.0)
    bw_mbs = (size_mb / (mean_io / 1000.0)) if mean_io > 0 else 0.0
    return {
        "runs": total_runs,
        "bytes": payload_bytes,
        "size_mb": round(size_mb, 3),
        "io_mean_ms": round(mean_io, 2),
        "io_p50_ms": round(float(np.percentile(io_arr, 50)), 2),
        "io_p95_ms": round(float(np.percentile(io_arr, 95)), 2),
        "bandwidth_mbs": round(bw_mbs, 2),
        "total_mean_ms": round(mean_tot, 2),
        "total_p95_ms": round(float(np.percentile(tot_arr, 95)), 2),
        "integrity_pct": round(100.0 * integrity_ok / max(1, total_runs), 2),
    }


def extract_hashes(metadata: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(metadata, dict):
        resources = metadata.get("resources")
        if isinstance(resources, list):
            for item in resources:
                if isinstance(item, dict) and item.get("hash"):
                    hashes.append(str(item["hash"]))
                elif isinstance(item, str) and item:
                    hashes.append(item)
        if not hashes and metadata.get("hash"):
            hashes.append(str(metadata["hash"]))
    elif isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, dict) and item.get("hash"):
                hashes.append(str(item["hash"]))
            elif isinstance(item, str) and item:
                hashes.append(item)
    seen = set()
    unique: list[str] = []
    for h in hashes:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def build_payload_file(base_bytes: bytes, target_kb: int, out_path: Path) -> tuple[int, int]:
    """Empaqueta N copias contiguas de la placa real en bloques de ~1 MB."""
    plate_size = len(base_bytes)
    target_bytes = target_kb * 1024
    copies = 1 if target_bytes <= plate_size else (target_bytes + plate_size - 1) // plate_size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_copies = max(1, (1024 * 1024) // plate_size)
    chunk_bytes = base_bytes * chunk_copies
    remaining = copies
    with out_path.open("wb") as f:
        while remaining >= chunk_copies:
            f.write(chunk_bytes)
            remaining -= chunk_copies
        if remaining > 0:
            f.write(base_bytes * remaining)
    return copies, copies * plate_size


def verify_and_infer(
    recognizer: PlateRecognizer,
    raw_bytes: bytes | bytearray | memoryview,
    base_bytes: bytes,
    expected_bytes: int,
    expected_plate: str,
    pattern: str,
) -> bool:
    plate_size = len(base_bytes)
    mv = memoryview(raw_bytes)
    try:
        if mv.nbytes != expected_bytes:
            return False
        first_plate = bytes(mv[:plate_size])
        last_plate = bytes(mv[-plate_size:])
    finally:
        mv.release()

    if first_plate != base_bytes or last_plate != base_bytes:
        return False
    arr = np.frombuffer(first_plate, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return False
    res = recognizer.predict_from_bgr(decoded, pattern=pattern)
    sync_device(recognizer.device)
    return res["plate"] == expected_plate


def fetch_kax_cloud_segment(client: Kax, hash_id: str) -> bytes:
    data = client.download_from_cloud(hash_id)
    if isinstance(data, (bytes, bytearray, memoryview)) and len(data) > 0:
        return bytes(data)
    raw = client.getRes(hash_id)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"download_from_cloud('{hash_id}') no devolvió bytes ni cargó SHM.")
    return bytes(raw)


def run_buffer_experiment(
    recognizer: PlateRecognizer,
    img_bgr: np.ndarray,
    pattern: str,
    expected_plate: str,
    num_items: int,
    buffer_size: int,
    producer_delay_ms: float,
    intermittent: bool = False,
    burst_size: int = 20,
    pause_ms: float = 180.0,
) -> dict[str, Any]:
    buf: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=buffer_size)
    stats = {
        "max_occupancy": 0,
        "producer_waits_full": 0,
        "consumer_waits_empty": 0,
        "consumer_idle_ms": 0.0,
        "processed": 0,
        "integrity_ok": 0,
    }
    lock = threading.Lock()

    def producer():
        for i in range(num_items):
            if intermittent:
                if i > 0 and i % burst_size == 0:
                    time.sleep(pause_ms / 1000.0)
            elif producer_delay_ms > 0:
                time.sleep(producer_delay_ms / 1000.0)

            if buf.full():
                with lock:
                    stats["producer_waits_full"] += 1

            buf.put(img_bgr)
            occ = buf.qsize()
            with lock:
                if occ > stats["max_occupancy"]:
                    stats["max_occupancy"] = occ

        buf.put(None)

    def consumer():
        while True:
            was_empty = buf.empty()
            if was_empty:
                with lock:
                    stats["consumer_waits_empty"] += 1

            t_wait = time.perf_counter()
            item = buf.get()
            wait_ms = (time.perf_counter() - t_wait) * 1000.0
            if was_empty:
                with lock:
                    stats["consumer_idle_ms"] += wait_ms

            if item is None:
                buf.task_done()
                break

            res = recognizer.predict_from_bgr(item, pattern=pattern)
            sync_device(recognizer.device)
            with lock:
                stats["processed"] += 1
                if res["plate"] == expected_plate:
                    stats["integrity_ok"] += 1
            buf.task_done()

    t0 = time.perf_counter()
    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tp.start()
    tc.start()
    tp.join()
    tc.join()
    elapsed_s = time.perf_counter() - t0

    return {
        "items": num_items,
        "buffer_size": buffer_size,
        "elapsed_s": round(elapsed_s, 3),
        "effective_fps": round(stats["processed"] / max(elapsed_s, 1e-6), 1),
        "max_occupancy": stats["max_occupancy"],
        "producer_waits_full": stats["producer_waits_full"],
        "consumer_waits_empty": stats["consumer_waits_empty"],
        "consumer_idle_ms": round(stats["consumer_idle_ms"], 1),
        "processed": stats["processed"],
        "integrity_pct": round(100.0 * stats["integrity_ok"] / max(1, stats["processed"]), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark escalado: jerarquía de memoria (hasta GB) y Buffering"
    )
    parser.add_argument("--image", default="data/raw/imgs/val/00014_AN-07_6_YR.jpg")
    parser.add_argument("--weights", default="checkpoints/crnn_best.pt")
    parser.add_argument("--regex", default="es_uc3m")
    parser.add_argument("--sizes-kb", type=int, nargs="+", default=[12, 1024, 5120, 10240])
    parser.add_argument("--max-cloud-kb", type=int, default=5120, help="Límite en KB para probar Cloud")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10, help="Repeticiones por tamaño (RAM, Disco, SHM)")
    parser.add_argument("--cloud-runs", type=int, default=3, help="Repeticiones por tamaño en Kax Cloud")
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--buffer-items", type=int, default=100, help="Placas por escenario de buffer")
    parser.add_argument("--skip-buffer", action="store_true", help="Omite la fase de buffering")
    parser.add_argument("--registry-url", default=os.getenv("KAX_REGISTRY_URL"))
    parser.add_argument("--storage-url", default=os.getenv("KAX_STORAGE_URL"))
    parser.add_argument("--broker", default=os.getenv("KAX_BROKER"))
    parser.add_argument("--port", type=int, default=int(os.getenv("KAX_PORT", "1883")))
    parser.add_argument("--out", default="results/benchmark_memoria_buffer.json")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

    base_bytes = img_path.read_bytes()
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"No se pudo decodificar la imagen: {img_path}")

    pattern = resolve_regex(args.regex)
    recognizer = PlateRecognizer(weights_path=args.weights)

    ref = recognizer.predict_from_disk(img_path, pattern=pattern)
    expected_plate = ref["plate"]

    for _ in range(args.warmup):
        recognizer.predict_from_bgr(img_bgr, pattern=pattern)
    sync_device(recognizer.device)

    infer_only_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        recognizer.predict_from_bgr(img_bgr, pattern=pattern)
        sync_device(recognizer.device)
        infer_only_ms.append((time.perf_counter() - t0) * 1000.0)
    base_infer_ms = float(np.mean(infer_only_ms))

    results: dict[str, Any] = {
        "device": str(recognizer.device),
        "image": str(img_path),
        "single_plate_bytes": len(base_bytes),
        "expected_plate": expected_plate,
        "base_infer_mean_ms": round(base_infer_ms, 2),
        "warmup": args.warmup,
        "hierarchy_by_size": {},
        "buffering_by_size": {},
    }

    tmp_dir = ROOT / "data" / "tmp_bench_payloads"
    has_cloud = bool(args.registry_url and args.storage_url and args.broker)

    try:
        for idx, target_kb in enumerate(args.sizes_kb, start=1):
            label = format_size_label(target_kb)
            payload_path = tmp_dir / f"lote_{label}.bin"
            copies, payload_bytes = build_payload_file(base_bytes, target_kb, payload_path)

            print(
                f"\n[{idx}/{len(args.sizes_kb)}] Evaluando tamaño {label} "
                f"({payload_bytes:,} bytes | {copies:,} placas)...",
                flush=True,
            )

            size_res: dict[str, Any] = {
                "label": label,
                "plates_in_batch": copies,
                "bytes": payload_bytes,
                "levels": {},
            }

            # 1) RAM (se carga y libera exclusivamente en este bloque)
            ram_arr = np.fromfile(payload_path, dtype=np.uint8)
            io_l, tot_l, ok_c = [], [], 0
            for _ in range(args.runs):
                t0 = time.perf_counter()
                copied = ram_arr.copy()
                t_io = time.perf_counter()
                ok = verify_and_infer(
                    recognizer, copied.data, base_bytes, payload_bytes, expected_plate, pattern
                )
                t_end = time.perf_counter()
                io_l.append((t_io - t0) * 1000.0)
                tot_l.append((t_end - t0) * 1000.0)
                ok_c += int(ok)
                del copied
            del ram_arr
            gc.collect()
            size_res["levels"]["ram"] = summarize_metrics(io_l, tot_l, payload_bytes, ok_c, args.runs)
            print("  - ram listo", flush=True)

            # 2) DISCO
            io_l, tot_l, ok_c = [], [], 0
            for _ in range(args.runs):
                t0 = time.perf_counter()
                raw = payload_path.read_bytes()
                t_io = time.perf_counter()
                ok = verify_and_infer(
                    recognizer, raw, base_bytes, payload_bytes, expected_plate, pattern
                )
                t_end = time.perf_counter()
                io_l.append((t_io - t0) * 1000.0)
                tot_l.append((t_end - t0) * 1000.0)
                ok_c += int(ok)
                del raw
            gc.collect()
            size_res["levels"]["disk"] = summarize_metrics(io_l, tot_l, payload_bytes, ok_c, args.runs)
            print("  - disk listo", flush=True)

            # 3) MEMORIA COMPARTIDA LOCAL (kax SHM)
            local_client = Kax(workers=1)
            local_hashes: list[str] = []
            try:
                meta_local = local_client.load_res(str(payload_path))
                local_hashes = extract_hashes(meta_local)
                if not local_hashes:
                    raise RuntimeError(f"Sin hashes en load_res: {meta_local}")

                io_l, tot_l, ok_c = [], [], 0
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    if len(local_hashes) == 1:
                        raw = local_client.getRes(local_hashes[0])
                    else:
                        raw = b"".join(bytes(local_client.getRes(h)) for h in local_hashes)
                    t_io = time.perf_counter()
                    ok = verify_and_infer(
                        recognizer, raw, base_bytes, payload_bytes, expected_plate, pattern
                    )
                    t_end = time.perf_counter()
                    io_l.append((t_io - t0) * 1000.0)
                    tot_l.append((t_end - t0) * 1000.0)
                    ok_c += int(ok)
                    del raw
                size_res["levels"]["shm_local"] = summarize_metrics(
                    io_l, tot_l, payload_bytes, ok_c, args.runs
                )
                size_res["levels"]["shm_local"]["segments"] = len(local_hashes)
                print(f"  - shm_local listo ({len(local_hashes)} seg)", flush=True)
            except OSError as exc:
                if exc.errno == errno.ENAMETOOLONG:
                    size_res["levels"]["shm_local"] = {"error": "Límite POSIX 31 chars en macOS"}
                else:
                    size_res["levels"]["shm_local"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  - shm_local error: {exc}", flush=True)
            except Exception as exc:
                size_res["levels"]["shm_local"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  - shm_local error: {exc}", flush=True)
            finally:
                for h in local_hashes:
                    try:
                        local_client.removeShm(h)
                    except Exception:
                        pass
                gc.collect()

            # 4) CLOUD (kax Cloud, solo hasta --max-cloud-kb)
            if has_cloud and target_kb <= args.max_cloud_kb:
                cloud_client = Kax(
                    registry_url=args.registry_url,
                    storage_url=args.storage_url,
                    broker=args.broker,
                    port=args.port,
                    workers=1,
                )
                cloud_hashes: list[str] = []
                uploaded = False
                try:
                    t_up0 = time.perf_counter()
                    meta_cloud = cloud_client.uploadToCloudFromDisk(str(payload_path))
                    upload_ms = (time.perf_counter() - t_up0) * 1000.0
                    cloud_hashes = extract_hashes(meta_cloud)
                    if not cloud_hashes:
                        raise RuntimeError(f"Sin hashes en uploadToCloudFromDisk: {meta_cloud}")
                    uploaded = True

                    io_l, tot_l, ok_c = [], [], 0
                    for _ in range(args.cloud_runs):
                        for h in cloud_hashes:
                            try:
                                cloud_client.removeShm(h)
                            except Exception:
                                pass
                        t0 = time.perf_counter()
                        if len(cloud_hashes) == 1:
                            raw = fetch_kax_cloud_segment(cloud_client, cloud_hashes[0])
                        else:
                            raw = b"".join(
                                fetch_kax_cloud_segment(cloud_client, h) for h in cloud_hashes
                            )
                        t_io = time.perf_counter()
                        ok = verify_and_infer(
                            recognizer, raw, base_bytes, payload_bytes, expected_plate, pattern
                        )
                        t_end = time.perf_counter()
                        io_l.append((t_io - t0) * 1000.0)
                        tot_l.append((t_end - t0) * 1000.0)
                        ok_c += int(ok)
                        del raw

                    cloud_summary = summarize_metrics(
                        io_l, tot_l, payload_bytes, ok_c, args.cloud_runs
                    )
                    cloud_summary["upload_ms"] = round(upload_ms, 2)
                    cloud_summary["segments"] = len(cloud_hashes)
                    size_res["levels"]["cloud_kax"] = cloud_summary
                    print(
                        f"  - cloud_kax listo ({len(cloud_hashes)} seg | upload={upload_ms:.1f} ms)",
                        flush=True,
                    )
                except Exception as exc:
                    size_res["levels"]["cloud_kax"] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"  - cloud_kax error: {exc}", flush=True)
                finally:
                    for h in cloud_hashes:
                        try:
                            cloud_client.removeShm(h)
                        except Exception:
                            pass
                        if uploaded:
                            try:
                                cloud_client.removeShmFromCloud(h)
                            except Exception:
                                pass
                    gc.collect()
            elif not has_cloud:
                size_res["levels"]["cloud_kax"] = {"skipped": "Faltan variables KAX_*"}
            else:
                size_res["levels"]["cloud_kax"] = {
                    "skipped": f"Supera --max-cloud-kb ({args.max_cloud_kb} KB)"
                }

            # Borra el archivo temporal de este tamaño antes de pasar al siguiente
            payload_path.unlink(missing_ok=True)
            gc.collect()
            results["hierarchy_by_size"][label] = size_res

        # 5) BUFFERING
        if not args.skip_buffer:
            print(
                f"\nEvaluando BUFFERING (N={args.buffer_items} placas/escenario, B={args.buffer_sizes})...",
                flush=True,
            )
            for b_size in args.buffer_sizes:
                key = f"B={b_size}"
                results["buffering_by_size"][key] = {
                    "equilibrado": run_buffer_experiment(
                        recognizer, img_bgr, pattern, expected_plate,
                        num_items=args.buffer_items, buffer_size=b_size,
                        producer_delay_ms=base_infer_ms, intermittent=False,
                    ),
                    "rafaga_intermitente": run_buffer_experiment(
                        recognizer, img_bgr, pattern, expected_plate,
                        num_items=args.buffer_items, buffer_size=b_size,
                        producer_delay_ms=0.0, intermittent=True,
                        burst_size=20, pause_ms=180.0,
                    ),
                    "productor_lento": run_buffer_experiment(
                        recognizer, img_bgr, pattern, expected_plate,
                        num_items=args.buffer_items, buffer_size=b_size,
                        producer_delay_ms=base_infer_ms * 2.5, intermittent=False,
                    ),
                }
                print(f"  - {key} listo", flush=True)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 96)
    print(
        f"JERARQUÍA DE MEMORIA POR TAMAÑO (dispositivo: {recognizer.device} | "
        f"inferencia base: {base_infer_ms:.2f} ms | placa: {expected_plate})"
    )
    print("=" * 96)
    for label, sdata in results["hierarchy_by_size"].items():
        print(f"\n--- Tamaño {label} ({sdata['bytes']:,} bytes | {sdata['plates_in_batch']:,} placas) ---")
        for lvl, v in sdata["levels"].items():
            if "io_mean_ms" in v:
                extra = f" | seg={v['segments']}" if "segments" in v else ""
                print(
                    f"  {lvl:10s} | runs={v['runs']:2d} | acceso={v['io_mean_ms']:8.2f} ms "
                    f"(p95={v['io_p95_ms']:8.2f}) | BW={v['bandwidth_mbs']:8.2f} MB/s | "
                    f"total+inf={v['total_mean_ms']:8.2f} ms | ok={v['integrity_pct']}%{extra}"
                )
            else:
                print(f"  {lvl:10s} | {v}")

    if results["buffering_by_size"]:
        print("\n" + "=" * 96)
        print(f"BUFFERING POR TAMAÑO DE BUFFER (N={args.buffer_items} placas | ráfaga=20 placas / pausa=180 ms)")
        print("=" * 96)
        for b_key, scenarios in results["buffering_by_size"].items():
            print(f"\n--- {b_key} ---")
            for esc, v in scenarios.items():
                print(
                    f"  {esc:20s} | tiempo={v['elapsed_s']:5.3f}s | {v['effective_fps']:5.1f} p/s | "
                    f"occ_max={v['max_occupancy']:2d}/{v['buffer_size']:2d} | "
                    f"esp_lleno={v['producer_waits_full']:2d} | esp_vacio={v['consumer_waits_empty']:2d} "
                    f"({v['consumer_idle_ms']:6.1f} ms) | ok={v['processed']}/{v['items']} ({v['integrity_pct']}%)"
                )
    print("=" * 96)
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
