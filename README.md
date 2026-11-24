# CRNN-CTC — Guía de Instalación y Ejecución

---

## Requisitos previos

* **Sistema operativo:** Ubuntu 24.04 / Linux
* **Python:** 3.12+
---

## Clonación e instalación

```bash
git clone [https://github.com/vanesgrim/CRNN---CTC.git](https://github.com/vanesgrim/CRNN---CTC.git)
cd CRNN---CTC

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Repositorio de imágenes

Descarga el dataset de matrículas recortadas (**UC3M-LP**) y coloca las imágenes respetaando que:

> Si solo vas a correr las pruebas y benchmarks sin reentrenar, basta con tener `checkpoints/crnn_best.pt` y al menos la imagen de referencia `data/raw/imgs/val/00014_AN-07_6_YR.jpg`.

---

## Entrenamiento del modelo

Con el entorno virtual activo y el dataset ubicado en `data/raw/`, ejecuta el script

```bash
python src/train.py
```

Al finalizar, los mejores pesos se guardan en `checkpoints/crnn_best.pt`.


```bash
export KAX_REGISTRY_URL="http://<IP_REGISTRY>:<PUERTO>"
export KAX_STORAGE_URL="http://<IP_STORAGE>:<PUERTO>"
export KAX_BROKER="<IP_BROKER>"
export KAX_PORT="1883"
```

---

## Ejecución de pruebas individuales y benchmarks

### A. Prueba individual de inferencia local
```bash
python src/infer.py --image data/raw/imgs/val/00014_AN-07_6_YR.jpg --weights checkpoints/crnn_best.pt
```

### B. Prueba individual de `kax` Cloud
```bash
python scripts/paso8b_prueba_kax_cloud.py
```

### C. Benchmark de jerarquía de memoria (12 KB a 10 MB) y *buffering* ($B = 2, 8, 32$)
```bash
python scripts/paso9_benchmark_memoria_buffer.py \
  --sizes-kb 12 1024 5120 10240 \
  --runs 20 \
  --cloud-runs 5 \
  --buffer-items 100 \
  --out results/benchmark_memoria_buffer.json
```


### D. Prueba de *streaming* local de 3 GB con *buffer pool* reutilizable (2 MB y 4 MB)
```bash
python scripts/paso10_streaming_buffer_pool.py \
  --total-gb 3.0 \
  --block-sizes-mb 2 4 \
  --pool-slots 2 8 \
  --pause-every 25 --pause-ms 80 \
  --skip-cloud \
  --out results/benchmark_3gb_pool_intermitente.json
```

### Prueba de reciclaje de cuota en `kax` Cloud 
```bash
python scripts/paso10_streaming_buffer_pool.py \
  --total-gb 0.01 --block-sizes-mb 2 --pool-slots 2 \
  --cloud-block-kb 980 --cloud-cycles 12 \
  --out results/benchmark_cloud_reciclaje.json
```

---


Todos los resultados se guardan automáticamente en formato JSON dentro de `results/`:

* `results/benchmark_memoria_buffer.json`
* `results/benchmark_3gb_pool_intermitente.json`
* `results/benchmark_cloud_reciclaje.json`