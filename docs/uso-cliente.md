# Embedding Service: guía de integración

Para quien consume el servicio desde otra aplicación por HTTP. Aquí está todo lo que hace falta para integrarlo:
autenticación, formatos exactos de petición y respuesta, límites, errores y buenas prácticas. Para instalarlo o
configurarlo, mira el [README](../README.md).

Valores a confirmar con quien opera el servicio:

| Marcador | Qué es |
| --- | --- |
| `<BASE_URL>` | URL del servicio desplegado (sin barra final) |
| `<API_KEY>` | Tu clave de acceso al servicio |
| `<MODEL>` | Alias del modelo a usar (ver [Modelo](#modelo)) |

## Resumen

El servicio convierte textos en vectores (*embeddings*) con el modelo que elijas. Tiene dos usos, con dos endpoints:

| Quiero... | Endpoint | Entrada | Respuesta |
| --- | --- | --- | --- |
| Vectorizar **preguntas de búsqueda** | `POST /embed/query` | `texts`: de 1 a 64 textos | **Síncrona**: los vectores vienen en la respuesta |
| **Indexar documentos** (se trocean en fragmentos) | `POST /embed/jobs` | `documents`: de 1 a 100 documentos | **Asíncrona**: devuelve un `job_id` y se consulta después |
| Ver qué modelos hay | `GET /models` | | Síncrona |

**Aceptan lotes siempre**: se envía una lista, aunque tenga un solo elemento. No hay un modo de "texto único".

Cada endpoint aplica el prefijo adecuado del modelo, si lo necesita: el de *pregunta* en `/embed/query` y el de
*documento* en `/embed/jobs`. No tienes que añadir nada al texto.

## Autenticación

Todos los endpoints, salvo `GET /` (health check, devuelve `{"status": "ok"}`), exigen una clave:

```
Authorization: Bearer <API_KEY>
```

Sin clave, con otro esquema (`Basic`, `Token`...) o con una clave incorrecta, responde `401`:

```json
{ "detail": "Invalid or missing API key" }
```
con la cabecera `WWW-Authenticate: Bearer`. Usa siempre HTTPS: la clave viaja en una cabecera.

## `POST /embed/query`: consultas de búsqueda

```http
POST <BASE_URL>/embed/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```
```json
{
  "texts": ["¿qué riesgos tiene la IA en medicina?", "otra pregunta"],
  "model": "<MODEL>"
}
```

Respuesta `200`, con un vector por texto **en el mismo orden**:
```json
{
  "count": 2,
  "dimensions": 1024,
  "model": "<MODEL>",
  "embeddings": [[0.0123, -0.0456, "..."], [0.0789, 0.0123, "..."]]
}
```

- De 1 a 64 textos. Cada texto es una cadena no vacía; los espacios al principio y al final se eliminan.
- **No trocea.** Cada texto debe caber en el contexto del modelo. Pensado para preguntas o frases cortas.
- Tarda lo que tarde el modelo o su proveedor: normalmente menos de un par de segundos.

## `POST /embed/jobs`: indexar documentos

El servicio trocea cada documento en fragmentos (por párrafos, luego frases, luego palabras, con un pequeño solape entre
fragmentos) y devuelve un vector por fragmento.

### 1. Crear el job

```http
POST <BASE_URL>/embed/jobs
Authorization: Bearer <API_KEY>
Content-Type: application/json
```
```json
{
  "documents": ["Texto completo del documento 1...", "Texto completo del documento 2..."],
  "model": "<MODEL>"
}
```

Respuesta `202`, inmediata:
```json
{ "job_id": "5a7a5e6271de4629a2d689c72fb9ab46", "status": "pending", "model": "<MODEL>" }
```

Límites (los valores actuales del servicio; si cambian, la respuesta `413` indica el máximo):

| Límite | Valor |
| --- | --- |
| Documentos por job | 100 |
| Caracteres por documento | 2.000.000 |
| Caracteres por petición (suma) | 10.000.000 |
| Jobs pendientes o en curso a la vez | 20 |

### 2. Consultar el job

```http
GET <BASE_URL>/embed/jobs/{job_id}
Authorization: Bearer <API_KEY>
```

Mientras trabaja (`pending` o `running`):
```json
{
  "job_id": "5a7a5e62...", "status": "running", "model": "<MODEL>",
  "progress": { "done": 8, "total": 21 },
  "dimensions": null, "count": null, "chunks": null, "error": null
}
```

Al terminar (`done`):
```json
{
  "job_id": "5a7a5e62...", "status": "done", "model": "<MODEL>",
  "progress": { "done": 21, "total": 21 },
  "dimensions": 1024,
  "count": 21,
  "chunks": [
    { "document_index": 0, "chunk_index": 0, "text": "Texto del primer fragmento...", "embedding": [0.012, -0.045, "..."] },
    { "document_index": 0, "chunk_index": 1, "text": "Texto del segundo fragmento...", "embedding": [0.031, 0.007, "..."] },
    { "document_index": 1, "chunk_index": 0, "text": "Fragmento del documento 2...", "embedding": [0.019, -0.002, "..."] }
  ],
  "error": null
}
```

Si falla (`failed`), `chunks` es `null` y `error` trae el motivo:
```json
{ "job_id": "5a7a5e62...", "status": "failed", "model": "<MODEL>", "progress": { "done": 0, "total": 0 }, "error": "Model '<MODEL>' is unavailable (ConnectError)" }
```

- `document_index` es la posición del documento en tu lista `documents`; `chunk_index`, la del fragmento dentro de ese
  documento. Con ambos reconstruyes de qué documento viene cada vector.
- `progress.total` puede ser `0` hasta que termine el primer lote de fragmentos.
- `text` es el texto exacto del fragmento que se vectorizó (sin el prefijo del modelo). Guárdalo junto al vector si
  quieres mostrarlo en los resultados de búsqueda.
- Un job **falla entero**: no hay resultados parciales.
- Los jobs se procesan **de uno en uno, por orden de llegada**.

### Cómo esperar el resultado

No hay conexión larga ni webhooks: se consulta periódicamente (*polling*).

- Espera 1-2 segundos entre consultas, aumentándolo hasta 5-10 s si el job tarda.
- Pon un tope total (por ejemplo 10 minutos) y trata el exceso como un error.
- Cada llamada es corta: timeouts de 10-30 s en tu cliente HTTP.

### Retención del resultado

Los jobs viven **en la memoria del servicio**. El resultado se conserva alrededor de una hora tras terminar y **se pierde
si el servicio se reinicia**. Guarda los vectores en tu base de datos en cuanto el job llegue a `done`. Si la consulta
devuelve `404`, vuelve a enviar el job.

## `GET /models`: modelos disponibles

```http
GET <BASE_URL>/models
Authorization: Bearer <API_KEY>
```
```json
{
  "default": "<MODELO_POR_DEFECTO>",
  "models": [
    { "key": "<MODEL>", "name": "baai/bge-m3", "provider": "openai", "default": true,
      "downloaded": null, "loaded": null, "dimensions": null, "metadata": {} }
  ]
}
```
`key` es el alias que se pasa en el campo `model`.

## Modelo

- El campo `model` es **opcional** en `/embed/query` y en `/embed/jobs`. Sin él se usa el modelo por defecto del servicio
  (`default` en `GET /models`).
- La respuesta incluye siempre el alias que se usó (`model`).
- **Nombre del modelo a usar: `<MODEL>` (pendiente de confirmar).**
- Un alias desconocido devuelve `422` con la lista de los válidos.
- **Cada modelo genera vectores de distinta dimensión** (por ejemplo 1024), y los vectores de modelos distintos **no se
  pueden comparar entre sí**. Indexa y consulta una misma colección siempre con el mismo modelo, y ajusta el tamaño de
  la columna de vectores (por ejemplo `vector(1024)` en pgvector) a las `dimensions` que devuelve el servicio. Si
  cambias de modelo, hay que reindexar.

## Errores

| Código | Cuándo | Qué hacer |
| --- | --- | --- |
| `401` | Falta la clave, el esquema no es `Bearer` o la clave no es válida | Revisar `Authorization` |
| `404` | El job no existe o caducó | Reenviar el job |
| `413` | Demasiados textos, texto demasiado largo o petición demasiado grande | Dividir en lotes menores |
| `422` | Lista vacía, texto vacío o solo espacios, o modelo desconocido | Corregir la petición |
| `429` | Hay demasiados jobs en cola | Reintentar con espera creciente |
| `503` | El modelo no está disponible, o su proveedor falla o rechaza la petición | Reintentar con espera; `detail` explica el motivo |

Casi todos los errores devuelven `{"detail": "<mensaje>"}` con un texto. **La excepción son los `422` por una petición mal
formada** (lista vacía, texto vacío, campos que faltan): ahí `detail` es una **lista de objetos** que indica el campo
con el problema:

```json
{ "detail": [ { "type": "too_short", "loc": ["body", "texts"], "msg": "List should have at least 1 item after validation, not 0" } ] }
```
El `422` por modelo desconocido sí trae un texto: `"Unknown model 'x'. Available: a, b"`.

Mensajes habituales de `503` (y del campo `error` de un job `failed`):

| Mensaje | Significado |
| --- | --- |
| `Model '<alias>' is not available` | El modelo no está listo en el servidor. Avisar a quien lo opera |
| `Model '<alias>' authentication failed` | El proveedor rechazó la clave del servicio. Avisar a quien lo opera |
| `Model '<alias>' request was rejected (HTTP 400)` | El proveedor rechazó el contenido, a menudo por un texto demasiado largo |
| `Model '<alias>' is unavailable (...)` | El proveedor no respondió tras varios reintentos (límite de uso, caída o *timeout*) |
| `Model '<alias>' returned an unexpected response` | Respuesta inválida del proveedor |
| `Embeddings could not be generated` | Fallo interno al generar los vectores |

## Buenas prácticas

- **Indexa por lotes de documentos**: de 10 a 50 por job, en vez de un job por documento. Un job por documento llena la
  cola (máximo 20) y se procesa en serie. Tampoco hagas jobs gigantes: si falla uno, se pierde entero.
- **Usa el endpoint adecuado**: `/embed/query` para preguntas, `/embed/jobs` para documentos. Mezclarlos da vectores
  ligeramente distintos en modelos que usan prefijos distintos para cada caso.
- **Reintenta `429` y `503`** con espera creciente (1 s, 2 s, 4 s...) y un número máximo de intentos. Los errores `4xx`
  restantes no se arreglan reintentando.
- **Guarda los vectores** (y el `text` de cada fragmento) en tu base de datos al recibir `done`.
- **Respuestas grandes**: cada vector son cientos o miles de números. Un job con cientos de fragmentos puede devolver
  varios MB de JSON; ajusta el tamaño de tus lotes en consecuencia.
- **Un fragmento mide del orden de 1.500 caracteres** con el modelo actual (depende del modelo y de su configuración),
  con un pequeño solape con el siguiente.
- **No envíes datos que no deban salir de tu infraestructura** si el modelo se sirve por API de un tercero: los textos se
  envían a ese proveedor. Consúltalo con quien opera el servicio.
- **Guarda el `X-Request-ID`** que devuelve cada respuesta (una cadena corta como `67068c17`). Si algo falla, quien opera
  el servicio lo busca en sus logs y ve toda la historia de esa petición. Inclúyelo al reportar un problema, junto con la
  hora y el `job_id` si lo hay.

## Ejemplos

### curl

```bash
# Consulta
curl -X POST "$BASE_URL/embed/query" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["¿qué riesgos tiene la IA en medicina?"]}'

# Indexar documentos
curl -X POST "$BASE_URL/embed/jobs" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Texto largo del documento..."]}'

# Consultar el job
curl "$BASE_URL/embed/jobs/$JOB_ID" -H "Authorization: Bearer $API_KEY"
```

### Python

```python
import time

import requests

BASE_URL = "https://..."  # <BASE_URL>
HEADERS = {"Authorization": "Bearer <API_KEY>"}


def embed_query(texts: list[str]) -> list[list[float]]:
    response = requests.post(f"{BASE_URL}/embed/query", json={"texts": texts}, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_documents(documents: list[str], max_wait_seconds: int = 600) -> list[dict]:
    response = requests.post(f"{BASE_URL}/embed/jobs", json={"documents": documents}, headers=HEADERS, timeout=30)
    response.raise_for_status()
    job_id = response.json()["job_id"]

    deadline = time.time() + max_wait_seconds
    delay = 1.0
    while time.time() < deadline:
        job = requests.get(f"{BASE_URL}/embed/jobs/{job_id}", headers=HEADERS, timeout=30)
        job.raise_for_status()
        body = job.json()
        if body["status"] == "done":
            return body["chunks"]  # document_index, chunk_index, text, embedding
        if body["status"] == "failed":
            raise RuntimeError(body["error"])
        time.sleep(delay)
        delay = min(delay * 1.5, 10)

    raise TimeoutError(f"job {job_id} did not finish in {max_wait_seconds}s")
```
