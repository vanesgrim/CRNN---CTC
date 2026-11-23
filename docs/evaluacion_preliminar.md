# Evaluación Preliminar — Sistema ALPR basado en CRNN-CTC

## 1. Resumen del Modelo y Configuración

Se implementó y entrenó desde cero una arquitectura **CRNN (Convolutional Recurrent Neural Network)** con pérdida **CTC (Connectionist Temporal Classification)** para reconocimiento óptico de matrículas sin segmentación previa de caracteres.

* **Arquitectura:** Extractor convolucional tipo VGG de 5 bloques con `BatchNorm2d` en todas las etapas (colapso de altura $32 \rightarrow 1$ manteniendo $T = 32$ columnas) + BiLSTM de 2 capas (256 unidades ocultas por dirección) + capa lineal sobre 37 clases (36 alfanuméricos + símbolo blanco $\epsilon$).
* **Complejidad y tamaño:** 8,199,653 parámetros entrenables | Checkpoint `checkpoints/crnn_best.pt` (**31.32 MB**).
* **Dataset:** Recortes de matrículas UC3M-LP — **2,056** imágenes de entrenamiento (`data/raw/imgs/train`) y **491** de validación (`data/raw/imgs/val`, **2,458** caracteres reales), omitiendo caracteres difuminados por privacidad (`_`).
* **Preprocesamiento:** Escala de grises, manejo de placas de dos renglones ($w/h < 1.6$), redimensionado proporcional a 32×128 con relleno a la derecha y normalización en $[-1, 1]$.
* **Aumento de datos (solo entrenamiento):** Brillo/contraste ($p=0.4$), desenfoque gaussiano ($p=0.2$), desenfoque de movimiento ($p=0.2$), ruido gaussiano ($p=0.2$), rotación $\pm 5^\circ$ ($p=0.2$) y borrado aleatorio ($p=0.15$).
* **Entrenamiento:** 70 épocas, lote de 32 imágenes, optimizador Adam ($\eta_0 = 3 \times 10^{-4}$), planificador *Cosine Annealing*, recorte de gradiente (`max_norm=5.0`) y selección del mejor modelo por menor `ValCER`.
* **Decodificación y postprocesamiento:** Decodificación voraz CTC (*best path*) con validación mediante expresión regular (`es_uc3m`: `^[0-9]{1,4}[A-Z]{1,3}$`).

---

## 2. Resultados Cuantitativos en Validación

Evaluación ejecutada sobre las **491 placas** de `data/raw/imgs/val` en acelerador **Apple MPS** (respaldo completo en `results/eval_summary.json` y `results/eval_predictions.csv`).

### Métricas globales

| Métrica | Resultado | Detalle |
| :--- | :---: | :--- |
| **Exactitud por placa (*Exact Match*)** | **76.37%** | 375 / 491 placas 100% exactas |
| **CER (*Character Error Rate*)** | **9.85%** | $CER = \frac{S + D + I}{N} = \frac{178 + 60 + 4}{2458}$ |
| **Exactitud por carácter ($1 - CER$)** | **90.15%** | 2,216 / 2,458 caracteres correctos |
| **Cumplimiento de formato (Regex `es_uc3m`)** | **96.95%** | 476 / 491 predicciones válidas |

### Distribución de errores por placa (Distancia de Levenshtein)

| Distancia de edición | Placas | Porcentaje |
| :--- | :---: | :---: |
| **0 errores (exacta)** | 375 | 76.4% |
| **1 error** | 64 | 13.0% |
| **2 errores** | 18 | 3.7% |
| **3 o más errores** | 34 | 6.9% |

El **89.4%** de las matrículas evaluadas presenta 0 o 1 error de carácter.

### Desglose por geometría física de la placa

| Formato detectado | Exactas / Total | Exactitud |
| :--- | :---: | :---: |
| **1 renglón ($w/h \ge 1.6$)** | 366 / 460 | **79.57%** |
| **2 renglones ($w/h < 1.6$)** | 9 / 31 | **29.03%** |

### Calibración de confianza CTC

| Grupo | Confianza media | Confianza mediana |
| :--- | :---: | :---: |
| **Placas exactas (375)** | 0.8679 | **0.9427** |
| **Placas con error (116)** | 0.4794 | **0.4474** |

### Desempeño computacional (Latencia en MPS)

Medición por imagen individual tras 10 iteraciones de calentamiento (*warmup*) con sincronización explícita del dispositivo (`torch.mps.synchronize()`):

| Media | Mediana (p50) | Percentil p95 | Throughput |
| :---: | :---: | :---: | :---: |
| **5.43 ms** | **5.28 ms** | **7.17 ms** | **184.2 placas/s** |

---

## 3. Análisis de Errores y Hallazgos Clave

1. **Predominio de sustituciones visuales ($73.6\%$ de los errores):** De las 242 ediciones totales, 178 fueron sustituciones ($S$), 60 borrados ($D$, $24.8\%$) y apenas 4 inserciones ($I$, $1.7\%$). El modelo prácticamente no genera caracteres fantasma. Las confusiones más frecuentes corresponden a pares morfológicamente similares bajo baja resolución o desenfoque: `6→8` (5), `8→9` (4), `7→3` (4), `P→R` (4), `9→3` (4), `M→N` (3) y `9→6` (3).
2. **Borrados en secuencias repetidas:** Los caracteres omitidos con mayor frecuencia (`3`, `4`, `1`, `9`, `V`, `D`) coinciden en varios casos con caracteres duplicados consecutivos o separados por borrado de privacidad donde la decodificación voraz CTC tiende a colapsarlos en una sola instancia (ej. `141DD → 141D`, `444PC → 44PC`).
3. **Sesgo posicional hacia formato moderno:** Los errores de mayor severidad (`BFMA66 → 851G`, `SG188 → 96BJ`, `M075V → 623MH`, `M847L → 117HV`) ocurren mayoritariamente en matrículas antiguas/provinciales que inician con letras. Debido al fuerte desbalance del conjunto hacia el formato moderno (dígitos al inicio, letras al final), la BiLSTM sesga la secuencia inicial hacia dígitos.
4. **Subrepresentación de placas de 2 renglones:** El rendimiento cae de **79.57%** a **29.03%** en placas cuadradas/dos líneas debido a su baja frecuencia en entrenamiento (~6%) y a la sensibilidad del particionado horizontal fijo en `h // 2`.
5. **Viabilidad de umbral de rechazo:** La brecha de ~0.50 entre las medianas de confianza de aciertos (0.9427) y errores (0.4474) permite implementar un filtro de control de calidad en inferencia para descartar o escalar lecturas de baja confianza.

---

## 4. Mejoras Propuestas para la Versión Final

* **Generación de placas sintéticas:** Incorporar datos sintéticos balanceados para formatos provinciales (letras iniciales), placas de dos renglones y pares conflictivos (`6/8/9`, `P/R`, `M/N`, dígitos repetidos).
* **Decodificación *Beam Search*:** Sustituir o complementar la decodificación voraz con búsqueda por haz (*prefix beam search*) para reducir borrados en caracteres repetidos.
* **Umbral de confianza configurable:** Añadir rechazo automático por debajo de $\tau \approx 0.70$ en `src/infer.py`.