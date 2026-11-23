from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.dataset import VALID_EXTS, parse_label_from_filename
from src.infer import PlateRecognizer, resolve_regex


def sync_device(device: torch.device) -> None:
    """Sincroniza operaciones asíncronas de GPU (CUDA/MPS) para medir latencia real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def align_edits(target: str, pred: str) -> tuple[int, int, int, list[tuple[str, str]], list[str], list[str]]:
    """
    Alineación óptima de Levenshtein (target -> pred).
    Devuelve:
      (S, D, I, sustituciones[(real, pred)], borrados[real], inserciones[pred])
    """
    n, m = len(target), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Borrado (el modelo omitió target[i-1])
                dp[i][j - 1] + 1,        # Inserción (el modelo añadió pred[j-1])
                dp[i - 1][j - 1] + cost, # Coincidencia o sustitución
            )

    # Reconstrucción hacia atrás para extraer operaciones exactas
    i, j = n, m
    subs: list[tuple[str, str]] = []
    dels: list[str] = []
    ins: list[str] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and target[i - 1] == pred[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs.append((target[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels.append(target[i - 1])
            i -= 1
        else:
            ins.append(pred[j - 1])
            j -= 1

    subs.reverse()
    dels.reverse()
    ins.reverse()
    return len(subs), len(dels), len(ins), subs, dels, ins


def main():
    parser = argparse.ArgumentParser(description="Evaluación completa del modelo CRNN-CTC en validación")
    parser.add_argument("--val-dir", default="data/raw/imgs/val", help="Carpeta de imágenes de evaluación")
    parser.add_argument("--weights", default="checkpoints/crnn_best.pt", help="Ruta al checkpoint .pt")
    parser.add_argument("--regex", default="es_uc3m", help="Preset regex o patrón personalizado")
    parser.add_argument("--warmup", type=int, default=10, help="Iteraciones de calentamiento antes de medir tiempo")
    parser.add_argument("--out-dir", default="results", help="Carpeta para guardar JSON y CSV de resultados")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = resolve_regex(args.regex)
    recognizer = PlateRecognizer(weights_path=args.weights)

    # Cargar lista de muestras válidas
    samples: list[tuple[Path, str]] = []
    for path in sorted(val_dir.iterdir()):
        if path.suffix.lower() not in VALID_EXTS:
            continue
        label = parse_label_from_filename(path.name)
        if label:
            samples.append((path, label))

    if not samples:
        raise RuntimeError(f"No se encontraron imágenes válidas en {val_dir}")

    # Calentamiento del dispositivo (evita sesgar la latencia con la compilación inicial de kernels)
    first_img = cv2.imread(str(samples[0][0]))
    for _ in range(args.warmup):
        recognizer.predict_from_bgr(first_img, pattern=pattern)
    sync_device(recognizer.device)

    # Acumuladores
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    edit_dist_hist: Counter[int] = Counter()
    sub_counter: Counter[tuple[str, str]] = Counter()
    del_counter: Counter[str] = Counter()
    ins_counter: Counter[str] = Counter()
    by_lines = {False: {"total": 0, "exact": 0}, True: {"total": 0, "exact": 0}}

    total_S = total_D = total_I = total_N = 0
    exact_matches = 0
    regex_valid_count = 0

    for path, real in samples:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        sync_device(recognizer.device)
        t0 = time.perf_counter()
        res = recognizer.predict_from_bgr(img_bgr, pattern=pattern)
        sync_device(recognizer.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred = res["plate"]
        conf = float(res["confidence"])
        valid_fmt = bool(res["valid_format"])
        two_lines = bool(res["two_lines"])

        S, D, I, subs, dels, ins = align_edits(real, pred)
        dist = S + D + I
        is_exact = dist == 0

        total_S += S
        total_D += D
        total_I += I
        total_N += len(real)

        if is_exact:
            exact_matches += 1
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)

        if valid_fmt:
            regex_valid_count += 1

        by_lines[two_lines]["total"] += 1
        by_lines[two_lines]["exact"] += int(is_exact)

        edit_dist_hist[min(dist, 3)] += 1  # Agrupa 3 o más en la cubeta '3'
        sub_counter.update(subs)
        del_counter.update(dels)
        ins_counter.update(ins)
        latencies_ms.append(latency_ms)

        rows.append(
            {
                "filename": path.name,
                "real": real,
                "pred": pred,
                "exact_match": is_exact,
                "edit_distance": dist,
                "S": S,
                "D": D,
                "I": I,
                "confidence": round(conf, 4),
                "regex_valid": valid_fmt,
                "two_lines": two_lines,
                "latency_ms": round(latency_ms, 3),
            }
        )

    total_plates = len(rows)
    plate_acc = exact_matches / total_plates
    total_edits = total_S + total_D + total_I
    cer = total_edits / max(1, total_N)
    char_acc = max(0.0, 1.0 - cer)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat = float(lat_arr.mean())
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    num_params = sum(p.numel() for p in recognizer.model.parameters())
    ckpt_mb = Path(args.weights).stat().st_size / (1024 * 1024)

    summary = {
        "model": {
            "weights": str(args.weights),
            "checkpoint_mb": round(ckpt_mb, 2),
            "parameters": num_params,
            "device": str(recognizer.device),
        },
        "dataset": {
            "val_dir": str(val_dir),
            "total_plates": total_plates,
            "total_chars_N": total_N,
            "regex_preset": args.regex,
            "regex_pattern": pattern,
        },
        "metrics": {
            "plate_accuracy_pct": round(plate_acc * 100.0, 2),
            "exact_matches": exact_matches,
            "cer_pct": round(cer * 100.0, 2),
            "char_accuracy_pct": round(char_acc * 100.0, 2),
            "substitutions_S": total_S,
            "deletions_D": total_D,
            "insertions_I": total_I,
            "regex_valid_pct": round((regex_valid_count / total_plates) * 100.0, 2),
        },
        "error_distribution": {
            "0_errors_exact": edit_dist_hist[0],
            "1_error": edit_dist_hist[1],
            "2_errors": edit_dist_hist[2],
            "3_or_more_errors": edit_dist_hist[3],
        },
        "confidence": {
            "mean_correct": round(float(np.mean(conf_correct)), 4) if conf_correct else 0.0,
            "median_correct": round(float(np.median(conf_correct)), 4) if conf_correct else 0.0,
            "mean_wrong": round(float(np.mean(conf_wrong)), 4) if conf_wrong else 0.0,
            "median_wrong": round(float(np.median(conf_wrong)), 4) if conf_wrong else 0.0,
        },
        "by_plate_layout": {
            "single_line": {
                "total": by_lines[False]["total"],
                "exact": by_lines[False]["exact"],
                "accuracy_pct": round(
                    100.0 * by_lines[False]["exact"] / max(1, by_lines[False]["total"]), 2
                ),
            },
            "two_lines": {
                "total": by_lines[True]["total"],
                "exact": by_lines[True]["exact"],
                "accuracy_pct": round(
                    100.0 * by_lines[True]["exact"] / max(1, by_lines[True]["total"]), 2
                ),
            },
        },
        "latency": {
            "warmup_runs": args.warmup,
            "mean_ms": round(mean_lat, 2),
            "median_p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "min_ms": round(float(lat_arr.min()), 2),
            "max_ms": round(float(lat_arr.max()), 2),
            "throughput_plates_per_sec": round(fps, 1),
        },
        "top_confusions": [
            {"real": r, "pred": p, "count": c} for (r, p), c in sub_counter.most_common(10)
        ],
        "top_deletions": [{"char": ch, "count": c} for ch, c in del_counter.most_common(10)],
        "top_insertions": [{"char": ch, "count": c} for ch, c in ins_counter.most_common(10)],
    }

    # Guardar archivos de salida
    csv_path = out_dir / "eval_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "eval_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reporte legible en consola
    print("=" * 76)
    print("REPORTE DE EVALUACIÓN DEL MODELO CRNN-CTC")
    print("=" * 76)
    print(f"Modelo:      {args.weights} ({ckpt_mb:.2f} MB | {num_params:,} parámetros | {recognizer.device})")
    print(f"Evaluación:  {val_dir} ({total_plates} placas | {total_N} caracteres reales)")
    print("-" * 76)
    print(f"Exactitud por placa (Exact Match):  {plate_acc*100:6.2f}%  ({exact_matches}/{total_plates})")
    print(f"CER (Character Error Rate):         {cer*100:6.2f}%  (S={total_S}, D={total_D}, I={total_I} / N={total_N})")
    print(f"Exactitud por carácter (1 - CER):   {char_acc*100:6.2f}%")
    print(f"Cumplimiento Regex ({args.regex}):        {(regex_valid_count/total_plates)*100:6.2f}%  ({regex_valid_count}/{total_plates})")
    print("-" * 76)
    print("Distribución de distancia de edición por placa:")
    print(f"  0 errores (exacta):  {edit_dist_hist[0]:4d} ({edit_dist_hist[0]/total_plates*100:5.1f}%)")
    print(f"  1 error:             {edit_dist_hist[1]:4d} ({edit_dist_hist[1]/total_plates*100:5.1f}%)")
    print(f"  2 errores:           {edit_dist_hist[2]:4d} ({edit_dist_hist[2]/total_plates*100:5.1f}%)")
    print(f"  3+ errores:          {edit_dist_hist[3]:4d} ({edit_dist_hist[3]/total_plates*100:5.1f}%)")
    print("-" * 76)
    print("Desglose por formato físico detectado (w/h < 1.6):")
    sl = summary["by_plate_layout"]["single_line"]
    tl = summary["by_plate_layout"]["two_lines"]
    print(f"  1 renglón:   {sl['exact']:4d}/{sl['total']:<4d} ({sl['accuracy_pct']:5.2f}%)")
    print(f"  2 renglones: {tl['exact']:4d}/{tl['total']:<4d} ({tl['accuracy_pct']:5.2f}%)")
    print("-" * 76)
    print("Calibración de confianza CTC:")
    c = summary["confidence"]
    print(f"  Placas exactas:    media = {c['mean_correct']:.4f} | mediana = {c['median_correct']:.4f}")
    print(f"  Placas con error:  media = {c['mean_wrong']:.4f} | mediana = {c['median_wrong']:.4f}")
    print("-" * 76)
    print(f"Latencia por placa (tras {args.warmup} ejecuciones de warmup en {recognizer.device}):")
    print(
        f"  Media: {mean_lat:.2f} ms | Mediana (p50): {p50_lat:.2f} ms | "
        f"p95: {p95_lat:.2f} ms | Throughput: {fps:.1f} placas/s"
    )
    print("-" * 76)
    top_subs = ", ".join(f"{r}->{p} ({cnt})" for (r, p), cnt in sub_counter.most_common(8)) or "Ninguna"
    top_dels = ", ".join(f"{ch} ({cnt})" for ch, cnt in del_counter.most_common(8)) or "Ninguno"
    top_ins = ", ".join(f"{ch} ({cnt})" for ch, cnt in ins_counter.most_common(8)) or "Ninguna"
    print(f"Top sustituciones (real->pred): {top_subs}")
    print(f"Top borrados (omitidos):        {top_dels}")
    print(f"Top inserciones (extra):        {top_ins}")

    # Mostrar 6 casos con mayor error para análisis cualitativo
    peores = sorted([r for r in rows if not r["exact_match"]], key=lambda x: (-x["edit_distance"], -x["confidence"]))
    print("-" * 76)
    print("Ejemplos de errores más severos (para análisis cualitativo):")
    for r in peores[:6]:
        print(
            f"  {r['filename']:24s} | Real={r['real']:6s} -> Pred={r['pred'] or '(vacío)':6s} "
            f"| dist={r['edit_distance']} (S={r['S']},D={r['D']},I={r['I']}) | conf={r['confidence']:.3f}"
        )

    print("=" * 76)
    print(f"Archivos exportados:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()