# Guía: servicio de embeddings en local

Servicio que parte documentos grandes en fragmentos por tokens y devuelve un vector por fragmento. Soporta **varios
modelos a la vez**: cada petición puede elegir uno por su alias, y añadir un modelo es solo editar un JSON.

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

Hay tres sitios, cada uno con su función:

| Archivo | Qué contiene |
| --- | --- |
| `config/models.json` | Los modelos y su configuración propia (prefijos, tamaño de fragmento, batch, metadatos...). |
| `config/service.json` | El modelo por defecto, la carpeta de modelos, los límites y la cola de jobs. |
| `.env` | Solo secretos y rutas de despliegue: `HF_TOKEN`, `CONFIG_DIR`, `MODEL_DIR`. |

```bash
cp .env.example .env     # solo hace falta si usas HF_TOKEN
```

Ejemplo de `config/models.json`:

```json
{
  "e5-small": {
    "name": "intfloat/multilingual-e5-small",
    "query_prefix": "query: ",
    "passage_prefix": "passage: "
  },
  "bge-m3": {
    "name": "BAAI/bge-m3",
    "chunk_tokens": 512
  }
}
```

La clave (`e5-small`, `bge-m3`) es el alias que se usa en las peticiones. Solo `name` es obligatorio. Lo demás es
opcional y por defecto no aplica nada específico de ningún modelo (sin prefijos, `normalize: true`, etc.). Si te
equivocas al escribir un campo (por ejemplo `querry_prefix`), el servicio no arranca y te dice cuál.

`config/service.json` tiene `default_model`, que debe ser uno de los alias de `models.json`.

## 4. Descargar modelos (una sola vez)

```bash
python -m scripts.download_model                 # el modelo por defecto
python -m scripts.download_model --model bge-m3  # uno concreto
python -m scripts.download_model --all           # todos los de models.json
```

Se guardan en `./models` (no se versiona). Los que ya están se omiten (`--force` para descargar de nuevo). El servicio
carga los modelos siempre desde disco, sin red, y solo en el momento en que alguien los pide.

## 5. Arrancar

```bash
uvicorn app.main:app --reload
```

Documentación interactiva: http://127.0.0.1:8000/docs

## 6. Probar

Ver qué modelos hay:

```bash
curl http://127.0.0.1:8000/models
```

Documentos grandes (asíncrono). Sin `model` se usa el modelo por defecto:

```bash
curl -X POST http://127.0.0.1:8000/embed/jobs \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Texto largo del primer documento...", "Otro documento..."], "model": "bge-m3"}'
# -> {"job_id": "abc123...", "status": "pending", "model": "bge-m3"}

curl http://127.0.0.1:8000/embed/jobs/abc123...
# -> status: pending | running | done | failed, progress {done,total}, y al acabar los chunks con sus vectores
```

Consultas de búsqueda (síncrono, aplica el prefijo de query del modelo):

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

print(status["model"], status["count"], status["dimensions"])
```

## 7. Añadir un modelo

1. Añade su bloque a `config/models.json` (prefijos solo si el modelo los necesita).
2. `python -m scripts.download_model --model <alias>`
3. Reinicia el servicio. Ya aparece en `GET /models` y se puede pedir con `"model": "<alias>"`.

Los vectores de modelos distintos no se pueden mezclar ni comparar: cada colección debe indexarse y consultarse con
el mismo modelo. Si cambias de modelo, actualiza las columnas (`vector(N)`) y reindexa.

Para cambiar el modelo por defecto, edita `default_model` en `config/service.json`.

## 8. Problemas comunes

| Síntoma | Causa y solución |
| --- | --- |
| El servicio no arranca: "Invalid configuration" | Un JSON está mal formado, hay un campo mal escrito o `default_model` no está en `models.json`. El mensaje dice qué y dónde. |
| `422` "Unknown model" | El alias no existe. El mensaje lista los disponibles; también `GET /models`. |
| `503` / job `failed` con "Model 'x' is not available" | El modelo no está descargado. El log del servidor indica el comando exacto. |
| Los resultados de búsqueda son malos | Faltan `query_prefix` / `passage_prefix` en un modelo que los necesita (E5). |
| Mucha memoria | Baja `max_loaded_models` en `service.json`; se descartan los modelos menos usados. |
| `413` | Superas algún límite de `service.json`. |
| `429` | Hay demasiados jobs en cola (`jobs.max_pending`). Reintenta más tarde. |
| `404` al consultar un job | Caducó (`jobs.ttl_seconds`) o el servicio se reinició: los jobs viven en memoria. |

## 9. Docker

```bash
docker build -t embedding-service .
docker run -p 8000:8000 -v embedding-models:/models embedding-service
```

La imagen incluye la carpeta `config/` pero no los modelos ni el `.env`. El modelo por defecto se descarga la primera
vez en el volumen `/models` y se reutiliza después. Para usar tu propia configuración, móntala:
`-v "$(pwd)/config:/app/config:ro"`. Para un modelo privado, `-e HF_TOKEN=...`.
