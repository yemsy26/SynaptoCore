# Referencia de la API

Base: `http://localhost:8080` (puerto configurable con `SYNAPTOCORE_PORT`). El servidor escucha solo en `127.0.0.1`.

Todas las peticiones y respuestas usan JSON en UTF-8. Los errores devuelven el código HTTP correspondiente y este cuerpo:

```json
{ "error": "El campo 'text' no puede estar vacío", "code": 400 }
```

| Código | Significado |
|--------|-------------|
| `400` | Petición inválida |
| `404` | Recurso inexistente |
| `409` | Conflicto (por ejemplo, ya hay una investigación en curso) |
| `500` | Error interno |
| `503` | Servicio externo no disponible (por ejemplo, sin SearXNG) |

---

## Estado

### `GET /health`

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "node_count": 156,
  "edge_count": 217,
  "avg_weight": 0.51,
  "passage_count": 16,
  "version": "0.17.0"
}
```

| Campo | Descripción |
|-------|-------------|
| `node_count` | Conceptos en el grafo |
| `edge_count` | Relaciones entre conceptos |
| `avg_weight` | Peso medio de las relaciones (0–1) |
| `passage_count` | Pasajes de texto guardados |

---

## Conversación

### `POST /api/v1/chat`

Punto de entrada del panel web. Decide si el mensaje enseña algo o pregunta, y responde en streaming con [Server-Sent Events](https://developer.mozilla.org/es/docs/Web/API/Server-sent_events). `POST /api/v1/query_stream` es un alias.

**Petición**

```json
{ "prompt": "¿Qué es el grafeno?", "session_id": "opcional" }
```

Si omites `session_id`, se crea una sesión nueva; envía el mismo valor en los mensajes siguientes para mantener el hilo de la conversación.

**Eventos**

Cada evento es una línea `data: {...}`:

```text
data: {"status":"researching","query":"¿Qué es el grafeno?","session_id":"…","done":false,"intent":"query"}
data: {"token":"El grafeno es","session_id":"…","done":false,"intent":"query"}
data: {"token":" una lámina de carbono [1].","session_id":"…","done":false,"intent":"query"}
data: {"token":"","session_id":"…","done":true,"intent":"query","answered_from":"web","sources":[{"n":1,"title":"Grafeno - Wikipedia","url":"https://es.wikipedia.org/wiki/Grafeno"}],"learning":true,"tool":null,"reinforced_edges":0,"timings":{"plan_ms":420,"context_ms":35,"research_ms":1300}}
```

| Evento | Cuándo |
|--------|--------|
| `status: "researching"` | La memoria no cubre la pregunta y se va a buscar en la web |
| `token` | Fragmento de la respuesta |
| `done: true` | Fin de la respuesta, con su origen y sus fuentes |
| `error` | Fallo; cierra el stream |

Campos del evento final:

| Campo | Descripción |
|-------|-------------|
| `answered_from` | `memory`, `web`, `tool`, `general` (conocimiento general no verificado) o `conversation` |
| `sources` | Fuentes citadas como `[n]` en la respuesta: web o pasajes de la memoria |
| `learning` | `true` si se está aprendiendo de las fuentes en segundo plano |
| `tool` | Herramienta usada, si la hubo |
| `reinforced_edges` | Relaciones de la memoria reforzadas por esta respuesta |

Cuando el mensaje enseña algo (`intent: "learn"`), el primer evento confirma lo aprendido e incluye `triples_count`, `nodes_created`, `edges_created` y `passages_saved`.

### `POST /api/v1/query`

La misma consulta, con la respuesta completa en un solo JSON. Siempre responde como consulta (no aprende del mensaje).

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "¿A qué hora abre la sucursal del centro los sábados?"}'
```

```json
{
  "response": "La sucursal del centro abre los sábados de 10 a 14 h [1].",
  "session_id": "3b9e…",
  "answered_from": "memory",
  "sources": [{ "n": 1, "title": "Lo que me enseñaste", "url": "" }]
}
```

### `POST /api/v1/learn`

Enseña un texto. Se extraen sus relaciones, se añaden al grafo como conocimiento permanente y el texto se guarda como pasaje.

```bash
curl -X POST http://localhost:8080/api/v1/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "La sucursal del centro abre de lunes a viernes de 9 a 18 h y los sábados de 10 a 14 h."}'
```

```json
{
  "triples_extracted": 3,
  "nodes_created": 4,
  "edges_created": 3,
  "llm_attempts": 1,
  "passages_saved": 1
}
```

---

## Grafo

### `GET /api/v1/graph/all`

Instantánea del grafo: los 2 000 conceptos más recientes y las relaciones entre ellos.

```json
{
  "nodes": [
    { "id": "grafeno", "label": "concept", "name": "grafeno", "properties": {}, "created_at": 1792805962 }
  ],
  "edges": [
    {
      "id": "6f1c…", "source_id": "grafeno", "relation": "tiene_estructura", "target_id": "hexagonal",
      "weight": 0.66, "properties": {}, "created_at": 1792805962, "last_accessed": 1792805962,
      "origin": "researched", "source": "https://es.wikipedia.org/wiki/Grafeno"
    }
  ]
}
```

`origin` indica la procedencia de cada relación: `taught` (enseñada), `researched` (investigada con fuentes), `inferred` (deducida por el aprendizaje autónomo), `imported` (importada por lotes), `federated`, `system` o `unknown`.

### `GET /api/v1/graph/neighbors/{id}`

Relaciones salientes de un concepto, de mayor a menor peso.

```bash
curl http://localhost:8080/api/v1/graph/neighbors/grafeno
```

```json
{
  "node_id": "grafeno",
  "direction": "outgoing",
  "neighbors": [
    { "relation": "tiene_estructura", "target_id": "hexagonal", "target_name": "hexagonal", "weight": 0.66 }
  ]
}
```

---

## Investigación profunda

### `POST /api/v1/investigations`

Inicia una investigación en segundo plano. Devuelve `202 Accepted` con su estado inicial, `409` si ya hay una en curso y `503` si no hay SearXNG configurado.

```bash
curl -X POST http://localhost:8080/api/v1/investigations \
  -H "Content-Type: application/json" \
  -d '{"objective": "¿Quién descubrió la penicilina y en qué año?", "rounds": 3}'
```

| Campo | Descripción |
|-------|-------------|
| `objective` | Pregunta o tema (hasta 500 caracteres) |
| `rounds` | Rondas máximas, de 1 a 5 (por defecto 3) |

### `GET /api/v1/investigations`

Las investigaciones más recientes, resumidas: `id`, `objective`, `status`, `stage`, `created_at`, `updated_at`.

### `GET /api/v1/investigations/{id}`

Estado completo:

```json
{
  "id": "…",
  "objective": "¿Quién descubrió la penicilina y en qué año?",
  "status": "done",
  "max_rounds": 3,
  "round": 1,
  "stage": "Conclusión lista",
  "questions": ["…"],
  "sources": [{ "n": 1, "title": "…", "url": "…", "snippet": "…", "round": 1 }],
  "hypotheses": [
    { "text": "Alexander Fleming la descubrió en 1928", "round": 1, "verdict": "supported",
      "confidence": 0.9, "supporting": [1, 3], "contradicting": [], "reasoning": "…" }
  ],
  "conclusion": "… [1][3]",
  "learned": ["alexander_fleming→descubrió→penicilina"],
  "error": null,
  "created_at": 1792805000,
  "updated_at": 1792805019,
  "finished_at": 1792805019
}
```

- `status`: `running`, `done`, `cancelled`, `failed` o `interrupted` (el servidor se reinició mientras corría).
- `verdict`: `pending`, `supported`, `refuted` o `uncertain`. `supporting` y `contradicting` son números de fuente.

### `POST /api/v1/investigations/{id}/cancel`

Detiene la investigación en su siguiente paso y conserva lo reunido.

```json
{ "cancelled": true }
```

---

## Pasajes

### `GET /api/v1/passages?limit=50`

Pasajes guardados, los más recientes primero (`limit` entre 1 y 500; 50 por defecto).

```json
[
  {
    "id": "18aafb53-…",
    "text": "El grafeno es una lámina de carbono de un átomo de grosor, organizada en hexágonos…",
    "title": "Grafeno: ¿Qué es y para qué sirve?",
    "url": "https://e-ficiencia.com/grafeno-caracteristicas-y-aplicaciones/",
    "origin": "researched",
    "question": "¿Qué es el grafeno?",
    "created_at": 1792805962,
    "last_used": 1792806100,
    "uses": 2
  }
]
```

Los pasajes enseñados tienen `origin: "taught"`, sin título ni URL.

### `DELETE /api/v1/passages/{id}`

Borra un pasaje. Devuelve `204 No Content`, o `404` si no existe.

---

## Cerebros

### `GET /api/v1/brains`

Motores, modelos disponibles y los cerebros en uso.

```json
{
  "engines": [
    {"id": "ollama", "label": "Ollama", "kind": "ollama", "base_url": "http://localhost:11434", "online": true, "error": null},
    {"id": "lmstudio", "label": "LM Studio", "kind": "openai", "base_url": "http://localhost:1234", "online": false, "error": "..."}
  ],
  "models": [
    {"engine": "ollama", "engine_label": "Ollama", "model": "qwen3.5:2b-q4_K_M", "display": "qwen3.5:2b-q4_K_M",
     "params": "1.9B", "params_b": 1.9, "quantization": "Q4_K_M", "bits": 4.0, "context": null,
     "loaded": true, "reasoning": "unknown", "tier": "ligero"}
  ],
  "rapido":   {"choice": {"engine": "ollama", "model": "qwen3.5:2b-q4_K_M"}, "label": "Ollama · qwen3.5:2b-q4_K_M",
               "scope": {"tier": "ligero", "synthesis_tokens": 384, "prompt_passages": 3, "investigation_rounds": 3, "conclusion_tokens": 450}},
  "profundo": {"choice": {"engine": "ollama", "model": "qwen3.5:2b-q4_K_M"}, "label": "Ollama · qwen3.5:2b-q4_K_M",
               "scope": {"tier": "ligero", "synthesis_tokens": 384, "prompt_passages": 3, "investigation_rounds": 3, "conclusion_tokens": 450}}
}
```

`reasoning`: `absent` (no razona), `optional` (se puede apagar), `always` (razona siempre) o `unknown`. `tier`: `ligero`, `medio` o `avanzado`. Cada modelo trae también `size_bytes` (tamaño del archivo).

La respuesta incluye la memoria del equipo y si los cerebros en uso caben:

```json
"memory": {"ram_total": 34037112832, "ram_free": 16649162752, "vram_total": 8585740288, "vram_free": 8342470656,
           "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU"},
"fit": {"status": "justo", "need_bytes": 23729713659, "available_bytes": 24991633408,
        "message": "Caben justos: necesitan unos 22,1 GB de los 23,3 GB libres. El equipo irá lento mientras estén cargados.",
        "idle_loaded": []}
```

`fit.status`: `ok`, `justo`, `no_cabe` o `desconocido` (sin medir). Solo cuentan los modelos que aún no están cargados; `idle_loaded` lista los cargados que no son cerebros. La VRAM se mide con `nvidia-smi` (sin GPU NVIDIA sale `null`).

### `POST /api/v1/brains/fit`

Si unos cerebros cabrían en la memoria libre, sin cambiarlos. Mismo cuerpo que `PUT /api/v1/brains`; responde con un `fit` como el de arriba.

### `PUT /api/v1/brains`

Cambia los cerebros sin reiniciar y guarda la elección en `cerebros.json`. `tier` es opcional (sin él, el nivel automático). Si alguno de los dos no está disponible, no se cambia ninguno (400).

```json
{
  "rapido":   {"engine": "ollama", "model": "qwen3.5:2b-q4_K_M"},
  "profundo": {"engine": "bionic", "model": "qwen3.6-35b-a3b", "tier": "avanzado"}
}
```

Responde como `GET /api/v1/brains`.

### `POST /api/v1/brains/test`

Prueba un modelo con un JSON con esquema y una respuesta en streaming. Si no está cargado, la prueba incluye la carga.

```json
{"engine": "bionic", "model": "qwen3.6-35b-a3b"}
```

```json
{
  "engine": "bionic", "model": "qwen3.6-35b-a3b", "tier": "avanzado",
  "json_ok": true, "json_detail": "{\"ok\": true, \"color\": \"azul\"}",
  "stream_ok": true, "first_token_ms": 5818, "tokens_per_s": 13.4,
  "sample": "La memoria a largo plazo es...", "total_ms": 31979,
  "verdict": "apto", "notes": []
}
```

`verdict`: `apto`, `apto con límites` o `no apto`; `notes` explica los límites.

## Federación

### `POST /api/v1/federation/export`

Devuelve un delta con los conceptos y relaciones de las últimas 24 horas, para guardarlo como archivo `.synapse`:

```json
{
  "delta_id": "a3f2c1d4-…",
  "agent_id": "synaptocore-…",
  "created_at": 1792805962,
  "schema_version": 1,
  "operations": [
    { "op": "UpsertNode", "node": { "id": "grafeno", "label": "concept", "name": "grafeno", "properties": {}, "created_at": 1792805962 } },
    { "op": "UpsertEdge", "edge": { "id": "…", "source_id": "grafeno", "relation": "tiene_estructura", "target_id": "hexagonal", "weight": 0.66, "…": "…" } }
  ]
}
```

Las relaciones del delta tienen los mismos campos que en `GET /api/v1/graph/all`.

---

## Telemetría

### `GET /ws/subconscious`

WebSocket con la actividad en tiempo real del aprendizaje autónomo, de las investigaciones y del aprendizaje a partir de fuentes:

```json
{ "event_type": "Consolidation", "message": "✓ Consolidada: grafeno→tiene_estructura→hexagonal (w=0.665)", "timestamp": 1792805962 }
```

### `GET /`

Panel web incrustado en el binario.
