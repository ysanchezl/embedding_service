# 🚀 Guía: Ejecutar servicio de embeddings (BGE-M3) en local

## 🧠 1. Requisitos previos

- Python **3.12**
- pip actualizado
- (opcional) virtualenv

---

## 📦 2. Descargar y preparar el proyecto

```bash
unzip bge_embedding_service.zip
cd bge_embedding_service
```

---

## 🧪 3. Crear entorno virtual (recomendado)

```bash
python3.14 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

---

## 📥 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ La primera vez descargará el modelo BGE-M3, puede tardar.

---

## 🚀 5. Arrancar el servidor

```bash
uvicorn app.main:app --reload
```

Deberías ver:

```
Uvicorn running on http://127.0.0.1:8000
```

---

## 🌐 6. Probar el endpoint

### 🔹 Opción A — Swagger
Abrir en navegador:

http://127.0.0.1:8000/docs

---

### 🔹 Opción B — curl

```bash
curl -X POST http://127.0.0.1:8000/embed \
-H "Content-Type: application/json" \
-d '{
  "texts": [
    "factura cliente ACME",
    "pedido pendiente envío"
  ]
}'
```

---

### 🔹 Opción C — Python

```python
import requests

res = requests.post(
    "http://127.0.0.1:8000/embed",
    json={"texts": ["hola mundo", "test embeddings"]}
)

print(res.json())
```

---

## ⚡ 7. Resultado esperado

```json
{
  "embeddings": [
    [0.0123, -0.456, ...],
    [0.0789, 0.123, ...]
  ]
}
```

---

## ⚠️ 8. Problemas comunes

### ❌ Tarda mucho al arrancar
Descarga del modelo → normal

### ❌ Error con torch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### ❌ Uso alto de memoria
BGE-M3 requiere ~4GB RAM

---

## 🧭 9. Test rápido

Probar con batch:

```json
{
  "texts": ["texto 1", "texto 2", "texto 3", "texto 4"]
}
```

---

## 🚀 10. Siguientes pasos

- Dockerizar
- Deploy en Azure Container Apps
- Integrar con LangGraph / RAG

