# Guía: servicio de embeddings en local

Servicio que parte documentos grandes en fragmentos por tokens y devuelve un vector por fragmento. Es agnóstico
del modelo: cambiar de modelo es solo cambiar la configuración.

## 1. Requisitos

- Python 3.10 o superior
- Unos 2 GB de RAM libres con `intfloat/multilingual-e5-small`

## 2. Entorno e instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Si torch da problemas o pesa demasiado, instala la versión de CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Configuración

```bash
cp .env.example .env
```

Edita `.env`. Lo importante:

- `MODEL_NAME`: modelo de Hugging Face (por defecto `intfloat/multilingual-e5-small`).
- `QUERY_PREFIX` / `PASSAGE_PREFIX`: los modelos E5 los necesitan (`"query: "` y `"passage: "`, con comillas por el
  espacio final). Si cambias a un modelo que no los usa, déjalos vacíos.
- `HF_TOKEN`: solo para modelos privados. Nunca lo subas a git.

## 4. Descargar el modelo (una sola vez)

```bash
python -m scripts.download_model
```

Se guarda en `./models` (no se versiona). Las siguientes ejecuciones lo detectan y no vuelven a descargar. El
servicio carga el modelo siempre desde disco, sin red. Para forzar una nueva descarga: `--force`.

## 5. Arrancar

```bash
uvicorn app.main:app --reload
```

Documentación interactiva: http://127.0.0.1:8000/docs

## 6. Probar

Documentos grandes (asíncrono):

```bash
curl -X POST http://127.0.0.1:8000/embed/jobs \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Texto largo del primer documento...", "Otro documento..."]}'
# -> {"job_id": "abc123...", "status": "pending"}

curl http://127.0.0.1:8000/embed/jobs/abc123...
# -> status: pending | running | done | failed, progress {done,total}, y al acabar los chunks con sus vectores
```

Consultas de búsqueda (síncrono, aplica el prefijo de query):

```bash
curl -X POST http://127.0.0.1:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"texts": ["¿qué riesgos tiene la IA en medicina?"]}'
```

En Python:

```python
import time, requests

base = "http://127.0.0.1:8000"
job = requests.post(f"{base}/embed/jobs", json={"documents": ["texto largo..."]}).json()

while True:
    status = requests.get(f"{base}/embed/jobs/{job['job_id']}").json()
    if status["status"] in ("done", "failed"):
        break
    time.sleep(1)

print(status["count"], status["dimensions"])
```

## 7. Cambiar de modelo

1. Edita `MODEL_NAME`, los prefijos y `TRUST_REMOTE_CODE` en `.env`.
2. `python -m scripts.download_model`
3. Reinicia el servicio.

Cambia la dimensión del vector: actualiza las columnas (`vector(N)`) y reindexa los datos existentes.

## 8. Problemas comunes

| Síntoma | Causa y solución |
| --- | --- |
| `503` / job `failed` con "model could not be loaded" | El modelo no está descargado o `MODEL_NAME` es incorrecto. Mira el log del servidor: indica la ruta y el comando a ejecutar. |
| Los resultados de búsqueda son malos | Faltan `QUERY_PREFIX` / `PASSAGE_PREFIX` en un modelo E5. |
| `413` | Superas `MAX_DOCUMENTS`, `MAX_DOCUMENT_CHARS` o `MAX_TOTAL_CHARS`. |
| `429` | Hay demasiados jobs en cola (`MAX_PENDING_JOBS`). Reintenta más tarde. |
| `404` al consultar un job | Caducó (`JOB_TTL_SECONDS`) o el servicio se reinició: los jobs viven en memoria. |

## 9. Docker

```bash
docker build -t embedding-service .
docker run -p 8000:8000 -v "$(pwd)/.env:/app/.env:ro" -v embedding-models:/models embedding-service
```

La imagen no incluye el modelo ni el `.env`. El modelo se descarga la primera vez en el volumen `/models` y se
reutiliza en los siguientes arranques.
