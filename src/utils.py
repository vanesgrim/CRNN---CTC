from __future__ import annotations

import re
import torch
import torch.nn.functional as F

from src.dataset import IDX_TO_CHAR


def ctc_collapse(indices: list[int]) -> str:
    """
    Aplica la función B(pi) de CTC sobre una secuencia de índices (0 = blanco):
      1) Colapsa índices idénticos consecutivos.
      2) Elimina el índice 0 (símbolo blanco ε).
    """
    chars: list[str] = []
    prev: int | None = None

    for idx in indices:
        if idx != prev:
            if idx != 0 and idx in IDX_TO_CHAR:
                chars.append(IDX_TO_CHAR[idx])
            # Importante: actualizamos 'prev' incluso si idx == 0,
            # para que un blanco entre dos letras iguales (ej. 4, 0, 4) permita repetirlas.
            prev = idx

    return "".join(chars)


def greedy_decode(logits: torch.Tensor) -> list[tuple[str, float]]:
    """
    Decodificación voraz (best path) sobre logits de forma (N, T, C).
    Devuelve una lista de tuplas (texto_decodificado, confianza) por cada placa del lote.
    """
    probs = F.softmax(logits, dim=2)            # (N, T, C)
    max_probs, max_idxs = probs.max(dim=2)      # (N, T) cada uno

    results: list[tuple[str, float]] = []
    for sample_idxs, sample_probs in zip(max_idxs.tolist(), max_probs.tolist()):
        text = ctc_collapse(sample_idxs)

        # Confianza: producto de probabilidades máximas en los pasos no blancos
        non_blank_probs = [p for idx, p in zip(sample_idxs, sample_probs) if idx != 0]
        if non_blank_probs:
            conf = 1.0
            for p in non_blank_probs:
                conf *= p
        else:
            conf = 0.0

        results.append((text, conf))

    return results


def postprocess_plate(text: str, pattern: str | None = None) -> tuple[str, bool]:
    """
    Paso 8 del documento: limpia símbolos no alfanuméricos y valida
    opcionalmente contra una expresión regular de formato de matrícula.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if pattern is None:
        return cleaned, len(cleaned) > 0
    is_valid = re.fullmatch(pattern, cleaned) is not None
    return cleaned, is_valid


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcula el número mínimo de sustituciones, borrados e inserciones (S + D + I)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        curr_row = [i]
        for j, c2 in enumerate(s2, start=1):
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(
                    prev_row[j] + 1,        # borrado
                    curr_row[j - 1] + 1,    # inserción
                    prev_row[j - 1] + cost, # sustitución o coincidencia
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def evaluate_predictions(preds: list[str], targets: list[str]) -> dict[str, float]:
    """
    Calcula las métricas del documento sobre un conjunto de predicciones:
      - plate_acc: fracción de placas 100% exactas (0.0 a 1.0)
      - cer: Character Error Rate = (S + D + I) / N
    """
    if not targets:
        return {"plate_acc": 0.0, "cer": 0.0}

    exact_matches = sum(1 for p, t in zip(preds, targets) if p == t)
    total_edits = sum(levenshtein_distance(p, t) for p, t in zip(preds, targets))
    total_chars = sum(len(t) for t in targets)

    return {
        "plate_acc": exact_matches / len(targets),
        "cer": total_edits / max(1, total_chars),
    }