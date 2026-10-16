# Guía: servicio de embeddings

Servicio que parte documentos grandes en fragmentos por tokens y devuelve un vector por fragmento. Soporta **varios
modelos a la vez, locales o por API**: cada petición puede elegir uno por su alias, y añadir un modelo (o un endpoint
nuevo) es solo editar un JSON.

| Proveedor | Qué es | Memoria | Necesita |
| --- | --- | --- | --- |
| `openai` | Cualquier servidor con el protocolo `/embeddings` de OpenAI: OpenAI, **Groq**, Mistral, Ollama, vLLM... | ~60 MB | una clave de API |
| `sentence-transformers` (por defecto) | Un modelo de Hugging Face dentro del servicio | ~1 GB | `requirements-local.txt` y descargarlo |

El modelo por defecto es `openrouter-bge-m3` (BAAI bge-m3 a través de OpenRouter, por API). **Los textos que se envían a un modelo por API salen de tu
infraestructura**: revisa las condiciones de datos del proveedor antes de mandar información sensible.

## 1. Requisitos

- Python 3.10 o superior
- Solo con modelos por API: unos 100 MB de RAM. Con un modelo local (`e5-small`): ~1 GB.

## 2. Entorno e instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt          # solo modelos por API (ligero)
pip install -r requirements-local.txt    # además, modelos locales (instala torch)
```

Si torch pesa demasiado, instala la versión de CPU antes:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Configuración

Hay tres sitios, cada uno con su función:

| Archivo | Qué contiene |
| --- | --- |
| `config/models.json` | Los modelos y su configuración propia (proveedor, URL, prefijos, tamaño de fragmento, batch, metadatos...). |
| `config/service.json` | El modelo por defecto, la carpeta de modelos locales, los límites y la cola de jobs. |
| `.env` | Solo secretos y rutas: `SERVICE_API_KEYS` (las claves de tus clientes), las claves de API de los modelos (`OPENROUTER_API_KEY`...), `HF_TOKEN`, `CONFIG_DIR`, `MODEL_DIR`. |

```bash
cp .env.example .env     # y rellena SERVICE_API_KEYS y OPENROUTER_API_KEY
```

**El servicio no arranca** si falta `SERVICE_API_KEYS`, o si el modelo por defecto es de API y falta su clave. El mensaje dice
qué variable falta.

Ejemplo de `config/models.json` con un modelo por API y uno local:

```json
{
  "groq-nomic": {
    "provider": "openai",
    "name": "nomic-embed-text-v1_5",
    "base_url": "https://api.groq.com/openai/v1",
    "api_key_env": "GROQ_API_KEY",
    "query_prefix": "search_query: ",
    "passage_prefix": "search_document: ",
    "chunk_tokens": 512
  },
  "e5-small": {
    "name": "intfloat/multilingual-e5-small",
    "query_prefix": "query: ",
    "passage_prefix": "passage: "
  }
}
```

La clave del JSON (`groq-nomic`, `e5-small`) es el alias que se usa en las peticiones.

- **Modelo local:** solo `name` es obligatorio. Lo demás es opcional y neutro por defecto.
- **Modelo por API** (`"provider": "openai"`): obligatorios `name`, `base_url`, `api_key_env` y `chunk_tokens`.
  `api_key_env` es el **nombre** de la variable de entorno que guarda la clave; la clave nunca va en el JSON.
- Un campo de un proveedor en un modelo del otro, o mal escrito, hace que el servicio no arranque y te dice cuál.
- En modelos por API el tamaño de los fragmentos se **estima por caracteres** (no hay tokenizador local):
  `chars_per_token` (3 por defecto) lo ajusta, bájalo si la API rechaza fragmentos por largos.

`config/service.json` tiene `default_model`, que debe ser uno de los alias de `models.json`.

## 4. Descargar modelos locales (una sola vez)

Solo hace falta para modelos locales; los de API no descargan nada.

```bash
python -m scripts.download_model                 # el modelo por defecto (no hace nada si es de API)
python -m scripts.download_model --model e5-small  # uno concreto
python -m scripts.download_model --all           # todos los locales de models.json
```

Se guardan en `./models` (no se versiona). Los que ya están se omiten (`--force` para descargar de nuevo). El servicio
carga los modelos locales siempre desde disco, sin red, y solo cuando alguien los pide.

## 5. Autenticación

Todos los endpoints, salvo `GET /` (el health check), exigen una clave en la cabecera:

```
Authorization: Bearer <clave>
```

- Las claves válidas están en `SERVICE_API_KEYS`, separadas por comas. Lo mejor es una clave por cliente; para rotarlas,
  añade la nueva antes de quitar la vieja.
- Genera una con: `python -c "import secrets; print(secrets.token_urlsafe(32))"` (mínimo 16 caracteres).
- Sin clave o con una incorrecta, el servicio responde `401`. Las claves no se escriben en los logs.
- Para desarrollo local, `DISABLE_AUTH=true` desactiva la autenticación (y deja un aviso en el log). Nunca en producción.
- En Swagger (`/docs`) pulsa **Authorize** y pega la clave.
- Usa HTTPS en producción; Render lo da por defecto.

## 6. Arrancar

```bash
uvicorn app.main:app --reload
```

Documentación interactiva: http://127.0.0.1:8000/docs

## 7. Probar

Todos los ejemplos llevan la clave en `Authorization` (aquí en la variable `$CLAVE`).

Ver qué modelos hay:

```bash
curl http://127.0.0.1:8000/models -H "Authorization: Bearer $CLAVE"
```

Documentos grandes (asíncrono). Sin `model` se usa el modelo por defecto:

```bash
curl -X POST http://127.0.0.1:8000/embed/jobs \
  -H "Authorization: Bearer $CLAVE" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Texto largo del primer documento...", "Otro documento..."], "model": "groq-nomic"}'
# -> {"job_id": "abc123...", "status": "pending", "model": "groq-nomic"}

curl http://127.0.0.1:8000/embed/jobs/abc123... -H "Authorization: Bearer $CLAVE"
# -> status: pending | running | done | failed, progress {done,total}, y al acabar los chunks con sus vectores
```

Consultas de búsqueda (síncrono, aplica el prefijo de query del modelo):

```bash
curl -X POST http://127.0.0.1:8000/embed/query \
  -H "Authorization: Bearer $CLAVE" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["¿qué riesgos tiene la IA en medicina?"]}'
```

En Python:

```python
import time, requests

base = "http://127.0.0.1:8000"
headers = {"Authorization": "Bearer " + CLAVE}
job = requests.post(f"{base}/embed/jobs", json={"documents": ["texto largo..."]}, headers=headers).json()

while True:
    status = requests.get(f"{base}/embed/jobs/{job['job_id']}", headers=headers).json()
    if status["status"] in ("done", "failed"):
        break
    time.sleep(1)

print(status["model"], status["count"], status["dimensions"])
```

## 8. Añadir un modelo

**Un endpoint por API** (OpenAI, Groq, Mistral, Ollama, vLLM...):

1. Añade a `config/models.json` un bloque con `"provider": "openai"`:

   ```json
   "openai-small": {
     "provider": "openai",
     "name": "text-embedding-3-small",
     "base_url": "https://api.openai.com/v1",
     "api_key_env": "OPENAI_API_KEY",
     "chunk_tokens": 512
   }
   ```

2. Pon `OPENAI_API_KEY` en el entorno o en `.env`.
3. Reinicia. Para que sea el modelo por defecto, cambia `default_model` en `config/service.json`.

**Un modelo local:**

1. Añade su bloque a `models.json` (prefijos solo si el modelo los necesita) e instala `requirements-local.txt`.
2. `python -m scripts.download_model --model <alias>`
3. Reinicia el servicio.

Los vectores de modelos distintos no se pueden mezclar ni comparar: cada colección debe indexarse y consultarse con
el mismo modelo. Si cambias de modelo, actualiza las columnas (`vector(N)`) y reindexa.

## 9. Problemas comunes

| Síntoma | Causa y solución |
| --- | --- |
| El servicio no arranca: "SERVICE_API_KEYS is not set" | Define al menos una clave de 16 o más caracteres (o `DISABLE_AUTH=true` solo en local). |
| `401` "Invalid or missing API key" | Falta la cabecera `Authorization: Bearer <clave>` o la clave no está en `SERVICE_API_KEYS`. |
| El servicio no arranca: "needs the X environment variable" | Falta la clave del modelo por defecto. Defínela en el entorno o en `.env`. |
| El servicio no arranca: "Invalid configuration" | Un JSON está mal formado, hay un campo mal escrito o `default_model` no está en `models.json`. El mensaje dice qué y dónde. |
| `422` "Unknown model" | El alias no existe. El mensaje lista los disponibles; también `GET /models`. |
| `503` / job `failed` con "Model 'x' is not available" | Local: no está descargado (el log da el comando). API: falta la clave (el log nombra la variable). |
| `503` con "authentication failed" | La API rechazó la clave. Revisa la variable que indica el log. |
| `503` con "request was rejected (HTTP 400)" | La API rechazó la petición, a menudo por fragmentos demasiado largos: baja `chunk_tokens` o `chars_per_token`. El log guarda el motivo exacto. |
| `503` con "unavailable" | La API no respondió tras los reintentos (límite de uso `429`, caída o timeout). Reintenta más tarde. |
| "Local models are not available in this deployment" | El modelo es local pero no está instalado `requirements-local.txt` (o la imagen Docker se construyó sin `LOCAL_MODELS=true`). |
| Los resultados de búsqueda son malos | Faltan los prefijos que el modelo necesita (E5, Nomic...), o el modelo no es bueno en tu idioma: pruébalo con tus textos. |
| `413` | Superas algún límite de `service.json`. |
| `429` | Hay demasiados jobs en cola (`jobs.max_pending`). Reintenta más tarde. |
| `404` al consultar un job | Caducó (`jobs.ttl_seconds`) o el servicio se reinició: los jobs viven en memoria. |

## 10. Docker y despliegue

```bash
docker build -t embedding-service .
docker run -p 8000:8000 -e SERVICE_API_KEYS=... -e OPENROUTER_API_KEY=... embedding-service
```

Por defecto la imagen instala solo `requirements.txt` (modelos por API): es pequeña y usa unos 60 MB de RAM, así que cabe
en una instancia de 512 MB. Para modelos locales:

```bash
docker build --build-arg LOCAL_MODELS=true -t embedding-service-local .
docker run -p 8000:8000 -v embedding-models:/models embedding-service-local
```

La imagen incluye `config/` pero no los modelos locales ni el `.env`. Para tu propia configuración, móntala con
`-v "$(pwd)/config:/app/config:ro"`. En Render u otra plataforma, define las claves como variables de entorno del
servicio (`SERVICE_API_KEYS` y `OPENROUTER_API_KEY`), no en archivos.
