from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import resource
import shutil
import sys
import threading
import time
from multiprocessing import shared_memory
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


def get_rss_mb() -> float:
    """Devuelve la memoria RAM residente actual del proceso (MB) en Linux."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024.0, 2)
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage_kb / 1024.0, 2)


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


def fetch_kax_cloud_segment(client: Kax, hash_id: str) -> bytes:
    data = client.download_from_cloud(hash_id)
    if isinstance(data, (bytes, bytearray, memoryview)) and len(data) > 0:
        return bytes(data)
    raw = client.getRes(hash_id)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"download_from_cloud('{hash_id}') no devolvió bytes ni cargó SHM.")
    return bytes(raw)


def build_aligned_dataset(base_bytes: bytes, total_gb: float, out_path: Path) -> tuple[int, int]:
    """Construye el archivo de streaming en bloques de ~4 MB alineados a plate_size."""
    plate_size = len(base_bytes)
    target_bytes = int(total_gb * (1024 ** 3))
    total_plates = (target_bytes + plate_size - 1) // plate_size
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_plates = max(1, (4 * 1024 * 1024) // plate_size)
    chunk_bytes = base_bytes * chunk_plates
    remaining = total_plates

    with out_path.open("wb") as f:
        while remaining >= chunk_plates:
            f.write(chunk_bytes)
            remaining -= chunk_plates
        if remaining > 0:
            f.write(base_bytes * remaining)

    return total_plates, total_plates * plate_size


def read_exact_into(f, buf: bytearray | memoryview) -> int:
    """Llena el bloque reutilizable directamente sin crear copias intermedias."""
    total = 0
    with memoryview(buf) as mv:
        size = len(mv)
        while total < size:
            with mv[total:] as sub:
                n = f.readinto(sub)
            if not n:
                break
            total += n
    return total


def verify_slot_and_infer(
    buf: bytearray | memoryview,
    n_bytes: int,
    base_bytes: bytes,
    recognizer: PlateRecognizer,
    expected_plate: str,
    pattern: str,
    do_infer: bool,
) -> bool:
    plate_size = len(base_bytes)
    if n_bytes < plate_size or (n_bytes % plate_size) != 0:
        return False

    with memoryview(buf) as mv:
        first_plate = bytes(mv[:plate_size])
        last_plate = bytes(mv[n_bytes - plate_size : n_bytes])

    if first_plate != base_bytes or last_plate != base_bytes:
        return False

    if not do_infer:
        return True

    arr = np.frombuffer(first_plate, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return False
    res = recognizer.predict_from_bgr(decoded, pattern=pattern)
    sync_device(recognizer.device)
    return res["plate"] == expected_plate


def run_reusable_pool_stream(
    dataset_path: Path,
    base_bytes: bytes,
    recognizer: PlateRecognizer,
    expected_plate: str,
    pattern: str,
    backend: str,
    block_mb: int,
    pool_slots: int,
    infer_every: int = 1,
    pause_every: int = 0,
    pause_ms: float = 0.0,
) -> dict[str, Any]:
    """
    Procesa el archivo completo reutilizando siempre los mismos `pool_slots` bloques
    preasignados en RAM (`ram_pool`) o en `/dev/shm` (`shm_pool`).
    """
    plate_size = len(base_bytes)
    plates_per_block = max(1, (block_mb * 1024 * 1024) // plate_size)
    block_bytes = plates_per_block * plate_size

    shm_objs: list[shared_memory.SharedMemory] = []
    slots: list[Any] = []
    pid = os.getpid()

    rss_before_mb = get_rss_mb()

    try:
        if backend == "ram_pool":
            slots = [bytearray(block_bytes) for _ in range(pool_slots)]
        elif backend == "shm_pool":
            for i in range(pool_slots):
                name = f"crnn_{pid}_{block_mb}m_{pool_slots}_{i}"
                shm = shared_memory.SharedMemory(create=True, size=block_bytes, name=name)
                shm_objs.append(shm)
                slots.append(shm.buf)
        else:
            raise ValueError(f"Backend no soportado: {backend}")

        free_q: queue.Queue[int] = queue.Queue(maxsize=pool_slots)
        filled_q: queue.Queue[tuple[int, int, int] | None] = queue.Queue(maxsize=pool_slots)
        for i in range(pool_slots):
            free_q.put(i)

        stats = {
            "blocks_read": 0,
            "blocks_processed": 0,
            "inferences_run": 0,
            "bytes_processed": 0,
            "integrity_ok": 0,
            "producer_waits_full": 0,
            "producer_wait_ms": 0.0,
            "consumer_waits_empty": 0,
            "consumer_idle_ms": 0.0,
            "disk_io_ms": 0.0,
            "max_slots_in_use": 0,
            "peak_rss_mb": rss_before_mb,
            "error": None,
        }
        lock = threading.Lock()

        def producer():
            block_idx = 0
            try:
                with dataset_path.open("rb", buffering=0) as f:
                    while True:
                        was_full = free_q.empty()
                        t_w0 = time.perf_counter()
                        slot_id = free_q.get()
                        wait_ms = (time.perf_counter() - t_w0) * 1000.0
                        if was_full:
                            with lock:
                                stats["producer_waits_full"] += 1
                                stats["producer_wait_ms"] += wait_ms

                        t_io0 = time.perf_counter()
                        n_read = read_exact_into(f, slots[slot_id])
                        io_ms = (time.perf_counter() - t_io0) * 1000.0

                        if n_read == 0:
                            free_q.put(slot_id)
                            break

                        block_idx += 1
                        in_use = pool_slots - free_q.qsize()
                        with lock:
                            stats["blocks_read"] += 1
                            stats["disk_io_ms"] += io_ms
                            if in_use > stats["max_slots_in_use"]:
                                stats["max_slots_in_use"] = in_use

                        filled_q.put((slot_id, n_read, block_idx))

                        if pause_every > 0 and pause_ms > 0 and (block_idx % pause_every == 0):
                            time.sleep(pause_ms / 1000.0)
            except Exception as exc:
                with lock:
                    stats["error"] = f"Producer {type(exc).__name__}: {exc}"
            finally:
                filled_q.put(None)

        def consumer():
            try:
                while True:
                    was_empty = filled_q.empty()
                    t_w0 = time.perf_counter()
                    item = filled_q.get()
                    wait_ms = (time.perf_counter() - t_w0) * 1000.0
                    if was_empty:
                        with lock:
                            stats["consumer_waits_empty"] += 1
                            stats["consumer_idle_ms"] += wait_ms

                    if item is None:
                        filled_q.task_done()
                        break

                    slot_id, n_read, block_idx = item
                    try:
                        do_infer = (block_idx % infer_every == 0)
                        ok = verify_slot_and_infer(
                            slots[slot_id],
                            n_read,
                            base_bytes,
                            recognizer,
                            expected_plate,
                            pattern,
                            do_infer=do_infer,
                        )
                        cur_rss = get_rss_mb() if (block_idx % 64 == 0) else 0.0
                        with lock:
                            stats["blocks_processed"] += 1
                            stats["bytes_processed"] += n_read
                            if do_infer:
                                stats["inferences_run"] += 1
                            stats["integrity_ok"] += int(ok)
                            if cur_rss > stats["peak_rss_mb"]:
                                stats["peak_rss_mb"] = cur_rss
                    finally:
                        free_q.put(slot_id)
                        filled_q.task_done()
            except Exception as exc:
                with lock:
                    stats["error"] = f"Consumer {type(exc).__name__}: {exc}"

        t0 = time.perf_counter()
        tp = threading.Thread(target=producer)
        tc = threading.Thread(target=consumer)
        tp.start()
        tc.start()
        tp.join()
        tc.join()
        elapsed_s = time.perf_counter() - t0

        rss_after_mb = get_rss_mb()
        if rss_after_mb > stats["peak_rss_mb"]:
            stats["peak_rss_mb"] = rss_after_mb

        total_mb = stats["bytes_processed"] / (1024.0 * 1024.0)
        pool_mem_mb = (block_bytes * pool_slots) / (1024.0 * 1024.0)
        disk_bw_mbs = (total_mb / (stats["disk_io_ms"] / 1000.0)) if stats["disk_io_ms"] > 0 else 0.0
        eff_bw_mbs = (total_mb / elapsed_s) if elapsed_s > 0 else 0.0

        return {
            "backend": backend,
            "block_mb_target": block_mb,
            "block_bytes": block_bytes,
            "plates_per_block": plates_per_block,
            "pool_slots": pool_slots,
            "pool_footprint_mb": round(pool_mem_mb, 2),
            "bytes_processed": stats["bytes_processed"],
            "gb_processed": round(stats["bytes_processed"] / (1024.0 ** 3), 3),
            "blocks_processed": stats["blocks_processed"],
            "inferences_run": stats["inferences_run"],
            "elapsed_s": round(elapsed_s, 3),
            "effective_throughput_mbs": round(eff_bw_mbs, 2),
            "disk_read_bw_mbs": round(disk_bw_mbs, 2),
            "max_slots_in_use": stats["max_slots_in_use"],
            "producer_waits_full": stats["producer_waits_full"],
            "producer_wait_ms": round(stats["producer_wait_ms"], 1),
            "consumer_waits_empty": stats["consumer_waits_empty"],
            "consumer_idle_ms": round(stats["consumer_idle_ms"], 1),
            "integrity_pct": round(
                100.0 * stats["integrity_ok"] / max(1, stats["blocks_processed"]), 2
            ),
            "rss_before_mb": rss_before_mb,
            "rss_peak_mb": stats["peak_rss_mb"],
            "error": stats["error"],
        }
    finally:
        slots.clear()
        for shm in shm_objs:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        shm_objs.clear()
        gc.collect()


def run_cloud_quota_recycling(
    base_bytes: bytes,
    recognizer: PlateRecognizer,
    expected_plate: str,
    pattern: str,
    registry_url: str,
    storage_url: str,
    broker: str,
    port: int,
    tmp_dir: Path,
    block_kb: int = 980,
    cycles: int = 12,
) -> dict[str, Any]:
    """
    Sube bloques únicos < 1 MiB (1 solo chunk), los descarga a SHM local,
    verifica integridad + inferencia y libera la nube inmediatamente para reciclar cuota.
    """
    plate_size = len(base_bytes)
    tag_len = 64
    max_single_chunk = 1048576  # 1 MiB límite de 1 chunk en kax
    target_bytes = min(block_kb * 1024, max_single_chunk - tag_len - plate_size)
    plates_in_block = max(1, (target_bytes - tag_len) // plate_size)
    payload_size = plates_in_block * plate_size + tag_len

    client = Kax(
        registry_url=registry_url,
        storage_url=storage_url,
        broker=broker,
        port=port,
        workers=1,
    )

    upload_times: list[float] = []
    download_times: list[float] = []
    total_times: list[float] = []
    ok_cycles = 0
    completed_cycles = 0
    total_bytes_transferred = 0
    first_error: str | None = None

    block_body = base_bytes * plates_in_block
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for c in range(1, cycles + 1):
        tag = f"KAX_RECYCLE_CYCLE_{c:04d}_{int(time.time() * 1000)}".ljust(tag_len, "_")[:tag_len].encode("ascii")
        block_bytes = block_body + tag
        block_file = tmp_dir / f"cloud_recycle_{c:04d}.bin"
        block_file.write_bytes(block_bytes)

        cloud_hashes: list[str] = []
        uploaded = False
        try:
            t_cycle0 = time.perf_counter()
            t_up0 = time.perf_counter()
            meta = client.uploadToCloudFromDisk(str(block_file))
            up_ms = (time.perf_counter() - t_up0) * 1000.0
            cloud_hashes = extract_hashes(meta)
            if not cloud_hashes:
                raise RuntimeError(f"Sin hashes en ciclo {c}: {meta}")
            uploaded = True

            # Eliminar copia local para forzar descarga real desde el servidor remoto
            for h in cloud_hashes:
                try:
                    client.removeShm(h)
                except Exception:
                    pass

            t_dl0 = time.perf_counter()
            if len(cloud_hashes) == 1:
                raw = fetch_kax_cloud_segment(client, cloud_hashes[0])
            else:
                raw = b"".join(fetch_kax_cloud_segment(client, h) for h in cloud_hashes)
            dl_ms = (time.perf_counter() - t_dl0) * 1000.0

            ok = (
                len(raw) == payload_size
                and raw[:plate_size] == base_bytes
                and raw[payload_size - tag_len - plate_size : payload_size - tag_len] == base_bytes
                and raw[-tag_len:] == tag
            )
            if ok:
                arr = np.frombuffer(raw[:plate_size], dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    res = recognizer.predict_from_bgr(decoded, pattern=pattern)
                    sync_device(recognizer.device)
                    ok = res["plate"] == expected_plate
                else:
                    ok = False

            tot_ms = (time.perf_counter() - t_cycle0) * 1000.0
            upload_times.append(up_ms)
            download_times.append(dl_ms)
            total_times.append(tot_ms)
            completed_cycles += 1
            total_bytes_transferred += payload_size
            ok_cycles += int(ok)
            print(
                f"  - Cloud ciclo {c:2d}/{cycles} | {payload_size:,} B | "
                f"up={up_ms:7.1f} ms | dl={dl_ms:6.1f} ms | ok={ok}",
                flush=True,
            )
            del raw
        except Exception as exc:
            first_error = f"Ciclo {c}/{cycles} -> {type(exc).__name__}: {exc}"
            print(f"  - Cloud detenido en ciclo {c}: {exc}", flush=True)
            break
        finally:
            for h in cloud_hashes:
                try:
                    client.removeShm(h)
                except Exception:
                    pass
                if uploaded:
                    try:
                        client.removeShmFromCloud(h)
                    except Exception:
                        pass
            block_file.unlink(missing_ok=True)
            gc.collect()

    size_mb = payload_size / (1024.0 * 1024.0)
    mean_dl = float(np.mean(download_times)) if download_times else 0.0
    mean_up = float(np.mean(upload_times)) if upload_times else 0.0
    return {
        "block_bytes": payload_size,
        "block_mb": round(size_mb, 3),
        "plates_per_block": plates_in_block,
        "cycles_requested": cycles,
        "cycles_completed": completed_cycles,
        "total_mb_recycled": round(total_bytes_transferred / (1024.0 * 1024.0), 2),
        "upload_mean_ms": round(mean_up, 2),
        "download_mean_ms": round(mean_dl, 2),
        "download_p95_ms": round(float(np.percentile(download_times, 95)), 2) if download_times else 0.0,
        "download_bw_mbs": round((size_mb / (mean_dl / 1000.0)), 2) if mean_dl > 0 else 0.0,
        "cycle_total_mean_ms": round(float(np.mean(total_times)), 2) if total_times else 0.0,
        "integrity_pct": round(100.0 * ok_cycles / max(1, completed_cycles), 2),
        "error": first_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Streaming de 3 GB con buffer pool reutilizable (RAM y SHM) + reciclaje de cuota en kax Cloud"
    )
    parser.add_argument("--image", default="data/raw/imgs/val/00014_AN-07_6_YR.jpg")
    parser.add_argument("--weights", default="checkpoints/crnn_best.pt")
    parser.add_argument("--regex", default="es_uc3m")
    parser.add_argument("--total-gb", type=float, default=3.0, help="Tamaño total del flujo local en GB")
    parser.add_argument("--block-sizes-mb", type=int, nargs="+", default=[2, 4], help="Tamaños de bloque reutilizable (MB)")
    parser.add_argument("--pool-slots", type=int, nargs="+", default=[2, 8], help="Número de bloques B en el pool")
    parser.add_argument("--infer-every", type=int, default=1, help="Ejecutar inferencia CRNN cada K bloques")
    parser.add_argument("--pause-every", type=int, default=0, help="Simular pausa del productor cada K bloques (0=continuo)")
    parser.add_argument("--pause-ms", type=float, default=0.0, help="Duración de la pausa del productor (ms)")
    parser.add_argument("--cloud-block-kb", type=int, default=980, help="Tamaño de bloque < 1 MiB para reciclaje Cloud")
    parser.add_argument("--cloud-cycles", type=int, default=12, help="Ciclos de reciclaje en kax Cloud")
    parser.add_argument("--skip-cloud", action="store_true", help="Omite la prueba de reciclaje en Cloud")
    parser.add_argument("--registry-url", default=os.getenv("KAX_REGISTRY_URL"))
    parser.add_argument("--storage-url", default=os.getenv("KAX_STORAGE_URL"))
    parser.add_argument("--broker", default=os.getenv("KAX_BROKER"))
    parser.add_argument("--port", type=int, default=int(os.getenv("KAX_PORT", "1883")))
    parser.add_argument("--out", default="results/benchmark_3gb_pool.json")
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

    # Calentamiento
    for _ in range(10):
        recognizer.predict_from_bgr(img_bgr, pattern=pattern)
    sync_device(recognizer.device)

    tmp_dir = ROOT / "data" / "tmp_stream_3gb"
    dataset_path = tmp_dir / f"stream_{args.total_gb:.1f}gb.bin"

    results: dict[str, Any] = {
        "device": str(recognizer.device),
        "expected_plate": expected_plate,
        "total_gb_target": args.total_gb,
        "infer_every_blocks": args.infer_every,
        "producer_pause_every": args.pause_every,
        "producer_pause_ms": args.pause_ms,
        "local_streaming_pools": [],
        "cloud_quota_recycling": None,
    }

    try:
        print(f"Generando flujo binario contiguo de {args.total_gb:.1f} GB en disco...", flush=True)
        t_gen0 = time.perf_counter()
        total_plates, total_bytes = build_aligned_dataset(base_bytes, args.total_gb, dataset_path)
        gen_s = time.perf_counter() - t_gen0
        print(
            f"Listo: {total_bytes / (1024**3):.3f} GB ({total_plates:,} placas reales) en {gen_s:.2f} s\n",
            flush=True,
        )
        results["dataset_bytes"] = total_bytes
        results["dataset_plates"] = total_plates

        for block_mb in args.block_sizes_mb:
            for b_slots in args.pool_slots:
                for backend in ("ram_pool", "shm_pool"):
                    print(
                        f"Corriendo {backend:8s} | bloque={block_mb}MB | B={b_slots} "
                        f"(pool={block_mb * b_slots} MB máx)...",
                        flush=True,
                    )
                    res = run_reusable_pool_stream(
                        dataset_path=dataset_path,
                        base_bytes=base_bytes,
                        recognizer=recognizer,
                        expected_plate=expected_plate,
                        pattern=pattern,
                        backend=backend,
                        block_mb=block_mb,
                        pool_slots=b_slots,
                        infer_every=args.infer_every,
                        pause_every=args.pause_every,
                        pause_ms=args.pause_ms,
                    )
                    results["local_streaming_pools"].append(res)
                    print(
                        f"  -> {res['gb_processed']:.3f} GB en {res['elapsed_s']:.2f} s | "
                        f"efec={res['effective_throughput_mbs']:.1f} MB/s | "
                        f"disco={res['disk_read_bw_mbs']:.1f} MB/s | "
                        f"pool={res['pool_footprint_mb']:.1f} MB | "
                        f"RSS pico={res['rss_peak_mb']:.1f} MB | ok={res['integrity_pct']}%",
                        flush=True,
                    )

        # Borrar los 3 GB locales antes de iniciar la prueba de nube
        dataset_path.unlink(missing_ok=True)
        gc.collect()

        has_cloud = bool(args.registry_url and args.storage_url and args.broker)
        if not args.skip_cloud and has_cloud:
            print(
                f"\nProbando reciclaje de cuota en kax Cloud "
                f"({args.cloud_cycles} ciclos x ~{args.cloud_block_kb} KB < 1 MiB)...",
                flush=True,
            )
            results["cloud_quota_recycling"] = run_cloud_quota_recycling(
                base_bytes=base_bytes,
                recognizer=recognizer,
                expected_plate=expected_plate,
                pattern=pattern,
                registry_url=args.registry_url,
                storage_url=args.storage_url,
                broker=args.broker,
                port=args.port,
                tmp_dir=tmp_dir,
                block_kb=args.cloud_block_kb,
                cycles=args.cloud_cycles,
            )
        elif args.skip_cloud:
            results["cloud_quota_recycling"] = {"skipped": "Omitido con --skip-cloud"}
        else:
            results["cloud_quota_recycling"] = {"skipped": "Faltan variables de entorno KAX_*"}
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 108)
    print(f"RESUMEN STREAMING LOCAL ({results['dataset_bytes'] / (1024**3):.3f} GB | {results['dataset_plates']:,} placas)")
    print("=" * 108)
    for r in results["local_streaming_pools"]:
        print(
            f"  {r['backend']:8s} | blk={r['block_mb_target']}MB | B={r['pool_slots']} "
            f"(pool={r['pool_footprint_mb']:5.1f} MB) | tiempo={r['elapsed_s']:6.2f}s | "
            f"eff={r['effective_throughput_mbs']:6.1f} MB/s | disk={r['disk_read_bw_mbs']:7.1f} MB/s | "
            f"esp_lleno={r['producer_waits_full']:4d} | ocioso={r['consumer_idle_ms']:6.1f} ms | "
            f"RSS_pico={r['rss_peak_mb']:6.1f} MB | ok={r['integrity_pct']}%"
        )

    c = results["cloud_quota_recycling"]
    print("\n" + "=" * 108)
    print("RESUMEN RECICLAJE DE CUOTA EN KAX CLOUD (< 1 MiB por bloque)")
    print("=" * 108)
    if isinstance(c, dict) and "cycles_completed" in c:
        print(
            f"  bloques={c['cycles_completed']}/{c['cycles_requested']} | "
            f"bloque={c['block_mb']} MB ({c['plates_per_block']} placas) | "
            f"total_reciclado={c['total_mb_recycled']} MB | "
            f"upload={c['upload_mean_ms']} ms | dl={c['download_mean_ms']} ms "
            f"(p95={c['download_p95_ms']} ms, BW={c['download_bw_mbs']} MB/s) | "
            f"ok={c['integrity_pct']}%"
        )
        if c.get("error"):
            print(f"  aviso: {c['error']}")
    else:
        print(f"  {c}")
    print("=" * 108)
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
