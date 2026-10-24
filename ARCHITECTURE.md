# SynaptoCore — Referencia de Arquitectura Interna

**Audiencia**: ingenieros que van a modificar, extender o depurar el sistema. No es documentación de usuario.

---

## 1. Modelo de Concurrencia

SynaptoCore usa un único proceso con tres capas de concurrencia:

```
┌─────────────────────────────────────────────────────────┐
│ Proceso: synaptocore.exe                                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Tokio Runtime (multi-thread, features = "full") │   │
│  │                                                  │   │
│  │  ┌─────────────┐  ┌──────────────────────────┐  │   │
│  │  │  Axum HTTP  │  │  Subconscious spawn_blocking│  │   │
│  │  │  (async I/O)│  │  (tokio thread pool)      │  │   │
│  │  └──────┬──────┘  └─────────────┬────────────┘  │   │
│  │         │                       │                │   │
│  │         ▼                       ▼                │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │         AppState (Arc<T> para todo)          │ │   │
│  │  │  graph: Arc<Mutex<MemoryGraph>>              │ │   │
│  │  │  llm:   Arc<dyn LlmBackend+Send+Sync>        │ │   │
│  │  │  sessions: Arc<SessionManager>  (RwLock)     │ │   │
│  │  │  tool_registry: Arc<ToolRegistry>            │ │   │
│  │  │  events_tx: broadcast::Sender<String>        │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  LlamaCppBackend → llama.cpp (C++ staticlib)            │
│  MemoryGraph     → rusqlite (SQLite WAL, bundled)        │
└─────────────────────────────────────────────────────────┘
```

**Decisiones de diseño críticas:**

- `MemoryGraph` usa `Mutex` (no `RwLock`) porque SQLite bundled con WAL ya serializa escrituras internamente. Añadir `RwLock` sobre él solo agregaría contención sin beneficio.
- `SessionManager` usa `RwLock` porque las lecturas del historial (correferencia) son mucho más frecuentes que las escrituras (fin de turno). Beneficio real en alta concurrencia.
- `LlamaCppBackend` **no implementa Send/Sync** nativamente (limitación de llama.cpp). Se llama siempre desde `tokio::task::spawn_blocking()`, que lo ejecuta en el thread pool de Tokio fuera del runtime async.
- `ToolRegistry` es inmutable post-inicialización. No necesita sincronización en runtime.

---

## 2. Esquema SQLite

El esquema se crea en `MemoryGraph::create_schema` y se actualiza en `run_migrations` (migraciones idempotentes: una base antigua se abre sin perder datos). Resumen de las tablas:

| Tabla | Contenido |
|-------|-----------|
| `nodes` | Conceptos: `id` canónico, `label`, `name`, `properties` (JSON), `created_at`, `embedding` (BLOB `f32` little-endian) y `canon` (nombre normalizado e indexado). |
| `edges` | Relaciones `source_id → relation → target_id` con `weight`, `created_at`, `last_accessed`, `decayed_at`, `origin` (procedencia) y `source` (URLs, lote de importación…). Única por tripleta. |
| `passages` | Pasajes de texto con `title`, `url`, `origin`, `question`, `last_used`, `uses` y `embedding`. Texto único. |
| `passages_fts` | Índice FTS5 de los pasajes (`unicode61 remove_diacritics 2`). |
| `meta` | Metadatos, como el modelo que generó los embeddings (si cambia, se regeneran). |
| `gap_attempts` | Pares de conceptos que exploró el aprendizaje autónomo y su enfriamiento. |
| `imports` | Lotes importados, con su progreso para reanudar o deshacer. |
| `investigations` | Investigaciones profundas: estado completo en JSON. |

**Pragmas** aplicados en cada conexión:

```sql
PRAGMA journal_mode = WAL;     -- los lectores no bloquean al escritor
PRAGMA synchronous  = NORMAL;  -- fsync solo en checkpoints
PRAGMA cache_size   = -16384;  -- 64 MB de caché de páginas
PRAGMA foreign_keys = ON;
PRAGMA temp_store   = MEMORY;
```

**Índices vectoriales.** Los embeddings de los nodos y de los pasajes se cargan al arrancar en dos índices planos en RAM (`VectorIndex`): búsqueda exacta por producto escalar sobre vectores normalizados. SQLite sigue siendo la fuente de verdad.

---

## 3. Pipeline de extracción de conocimiento

```
texto libre
    │
    ▼
LlmBridge::extract_entity()
    │  system: prompts::extraction_system_prompt()
    │  user:   texto del usuario
    │  max_tokens: 512, temp: 0.1
    │
    ▼ JSON: {"triples": [{"subject", "relation", "object", "weight"}]}
    │
    ├─► MAX_PARSE_RETRIES = 2 (reintenta si el JSON es inválido)
    │
    ▼
EpistemicValidator::validate_triplet()
    │  Busca evidencia en SearXNG (hasta MAX_SEARCH_SNIPPETS fuentes)
    │  Pasa snippets al LLM: ¿Verificado / Contradictorio / Inconcluso?
    │
    ├─ Verificado   → weight tal cual (máx 1.0)
    ├─ Inconcluso   → weight *= 0.5 si > threshold, sino rechaza
    └─ Contradictorio → rechaza, no inserta
    │
    ▼
MemoryGraph::insert_node() + insert_edge()
    │  Genera embedding con fastembed (si feature activo)
    │  INSERT OR REPLACE con weight validado
    │
    ▼
SubconsciousEvent → broadcast → WebSocket /ws/subconscious
```

---

## 4. Bucle ReAct (Tool Use)

El `LlmBridge::query_with_memory()` implementa el patrón **Reason + Act**:

```
MAX_TOOL_ITERATIONS = 3
scratchpad = ""

loop:
    prompt = tool_decision_prompt(query_reescrita, tools.describe_all(), scratchpad)
    respuesta_llm = llm.complete(prompt)   ← JSON: {"tool": "X", "params": {...}}
                                             o      {"tool": "none"}

    si tool == "none" o iteraciones >= MAX_TOOL_ITERATIONS:
        break

    resultado = ToolRegistry::get(tool).execute(params)
    scratchpad += f"Ejecuté {tool} → {resultado}\n"
    iteraciones += 1

síntesis = llm.complete(synthesis_prompt(query, contexto_grafo, scratchpad))
respuesta → usuario
```

**Resiliencia ante JSON inválido**: si el LLM devuelve texto no-JSON (alucinación), el parser captura el error y asume `{"tool": "none"}` para evitar bucles infinitos.

**Herramientas registradas** (inicialización en `main.rs`):

| Nombre | Input JSON | Descripción |
|--------|-----------|-------------|
| `TimeTool` | `{}` | Fecha/hora actual del sistema (LLMs no tienen noción del tiempo) |
| `WebhookTool` | `{"url": "..."}` | GET HTTP a sistemas externos. Timeout 10s. |
| Conectores (`connectors.rs`) | los `params` declarados | Cualquier API HTTP declarada en `conectores.json` (Paso 25) |
| `Calculadora` (`math_tools.rs`) | `{"expresion": "..."}` | Cálculo exacto: enteros de cualquier tamaño, fracciones, primos, factorización, combinatoria (Paso 29) |
| `Conjetura` (`math_tools.rs`) | `{"afirmacion": "...", "desde": 1, "hasta": 1000}` | Comprueba una afirmación sobre n en un rango y da el primer contraejemplo (Paso 29) |
| `Sucesion` (`math_tools.rs`) | `{"terminos": "..."}` | Fórmula polinómica, geométrica o de recurrencia que confirmen los términos dados, y los siguientes (Paso 29) |

Conectar un sistema externo no requiere código: se declara en `conectores.json` (ver README, «Conectores externos»). Para una herramienta con lógica propia, implementar `trait Tool` y llamar `tool_registry.register(MiHerramienta)` en `main.rs`.

**Respuesta directa** (Paso 29): una herramienta cuyo `answers_directly()` devuelve `true` entrega su salida como respuesta, sin pasar por el modelo de síntesis (`PreparedAnswer.direct`). Las tres matemáticas lo hacen: al parafrasear, el 2B recalculaba las cifras y las estropeaba (convertía `a(n) = n^2 + n` en «el siguiente es 930»). La respuesta repite la expresión evaluada, así que un planteamiento erróneo del planificador (`C(8,1)` en lugar de `8!`) queda a la vista. El evaluador es propio (sin `eval`), con límites de tamaño (~120 000 cifras), pasos y tiempo (3 s; 6 s en conjeturas).

---

## 5. Curva del Olvido

Basada en la función exponencial de Ebbinghaus:

$$w(t) = w_0 \cdot e^{-\lambda t}$$

donde:
- $w_0$ = peso actual de la arista
- $\lambda$ = tasa de olvido de la **procedencia** de la arista (Paso 26)
- $t$ = segundos desde `last_accessed`

Cada arista guarda de dónde vino (`edges.origin`) y su fuente (`edges.source`: URLs, el par que exploró el subconsciente…). En producción `DecayPolicy::ByOrigin` fija λ a partir de una vida media:

| Procedencia | Quién la escribe | Vida media sin uso |
|-------------|------------------|--------------------|
| `taught` | `/learn` y la intención «aprender» del chat | no se olvida |
| `system` | conocimiento base sembrado al arrancar | no se olvida |
| `researched` | aprendizaje de la investigación en caliente (con URLs) | 90 días |
| `unknown` | aristas anteriores al Paso 26 | 90 días |
| `imported` | `synaptocore importar` (fuente = nombre del lote) | no se olvida; se retira con `--deshacer <lote>` |
| `federated` | otro agente, vía federación | 30 días |
| `inferred` | consolidación del subconsciente | 14 días |

Si una arista se vuelve a aprender con una procedencia más duradera (`learn_edge`), sube de categoría y su reloj se reinicia: lo que el subconsciente dedujo pasa a «investigado» cuando una fuente lo confirma, y a «enseñado» si lo enseña el usuario; nunca baja; lo enseñado también reemplaza a lo importado, para que deshacer un lote no lo borre. El importador nunca modifica una arista existente. `DecayPolicy::Uniform(λ)` aplica el mismo λ a todo (demos y tests). La pasada del olvido solo lee las aristas que decaen.

**Pasajes (Paso 30).** La misma pasada olvida los pasajes (`passages`: texto de las fuentes y de lo enseñado, ver `memory_graph/passages.rs`) con la tasa de su procedencia, contando desde `last_used`: un pasaje vale 1 cuando fundamenta una respuesta y se borra por debajo del umbral de poda (lo investigado, tras ~13 meses sin usarse; lo enseñado, nunca). Tienen su propio índice vectorial en RAM y un índice FTS5 (`passages_fts`) para la búsqueda híbrida.

**Refuerzo Hebbiano**: cuando un nodo se consulta exitosamente (`semantic_search` retorna hit), se llama `MemoryGraph::reinforce_edge()` que incrementa `weight` y resetea `last_accessed`. Implementa "las neuronas que se activan juntas, se conectan."

**Poda**: aristas con $w(t) < \text{prune\_threshold}$ (0.05 en producción) se eliminan en la limpieza horaria; lo que no se olvida nunca se poda.

---

## 6. Sistema de Prompts

Todos los prompts están centralizados en `src/prompts.rs` como funciones puras `fn(&str, ...) -> String`. Ningún prompt está hardcodeado en la lógica de negocio.

| Función | Propósito |
|---------|----------|
| `extraction_system_prompt()` | Extrae tripletas de texto libre |
| `coreference_system_prompt()` | Reescribe prompt con contexto de sesión |
| `tool_decision_prompt(query, tools, scratchpad)` | Decide qué herramienta usar (o ninguna) |
| `synthesis_prompt(query, context, scratchpad)` | Genera respuesta final |
| `curiosity_question_prompt(node_a, node_b)` | Genera hipótesis entre dos nodos |
| `validation_prompt(claim, snippets)` | Valida epistémicamente una tripleta |

Para ajustar el comportamiento del LLM sin tocar lógica: solo editar `prompts.rs`.

---

## 7. Telemetría WebSocket

El servidor mantiene un `broadcast::Sender<String>` (capacidad 128) que distribuye eventos a todos los clientes WebSocket conectados en `/ws/subconscious`.

Estructura de evento (JSON serializado):
```json
{"event_type": "string", "message": "string", "timestamp": 1234567890}
```

Tipos de evento emitidos:

| event_type | Fuente | Descripción |
|-----------|--------|-------------|
| `learn` | `handle_learn` | Texto aprendido (N tripletas extraídas) |
| `query` | `handle_query` | Consulta recibida |
| `sleep_cycle` | `CuriosityEngine` | Hipótesis validada y consolidada |
| `decay` | `ForgettingCurve` | Resultado del cleanup nocturno |
| `tool_use` | `LlmBridge` | Herramienta ejecutada en bucle ReAct |

Si el buffer de 128 mensajes se llena (cliente WS lento), los mensajes más antiguos se descartan (`broadcast::error::SendError` ignorado). Se prioriza la frescura de datos sobre la completitud histórica.

---

## 8. Memoria Federada — Formato .synapse

El protocolo de sincronización es **append-only + idempotente**:

- Cada operación tiene un `delta_id` UUID v4. Importar el mismo delta dos veces es idempotente (INSERT OR REPLACE).
- No hay negociación de versiones ni vector clocks. La última escritura gana por timestamp.
- Adecuado para sincronización eventual en redes de baja confiabilidad (WiFi de sucursal).

Operaciones soportadas en `DeltaOperation`:
- `UpsertNode(Node)` — crea o actualiza nodo
- `UpsertEdge { source_id, relation, target_id, weight }` — crea o actualiza arista
- `DeleteEdge { source_id, relation, target_id }` — elimina arista (raro, usado por poda)

---

## 9. Gestión de errores

El proyecto usa `thiserror` para errores tipados por dominio:

| Tipo | Archivo | Casos |
|------|---------|-------|
| `MemoryError` | `error.rs` | SQLite, migraciones, parseo de BLOB |
| `BridgeError` | `llm_bridge.rs` | LLM timeout, JSON inválido, carga de modelo |
| `ToolError` | `tools.rs` | Input inválido, red, ejecución |
| `FederationError` | `federated_memory.rs` | I/O de archivos .synapse, JSON |
| `CuriosityError` | `subconscious.rs` | Grafo pequeño, generación fallida |
| `ApiError` | `api_server.rs` | HTTP 400/404/500 para clientes REST |

La API REST convierte todos los errores internos a `ApiError` antes de serializar. Los errores de `ToolError` **nunca** se propagan al usuario como HTTP 500 — se convierten en strings legibles que el LLM puede interpretar en la síntesis de respuesta.

---

## 10. Añadir un nuevo módulo

Checklist para extender SynaptoCore:

```
1. Crear src/mi_modulo.rs
2. Añadir `mod mi_modulo;` en src/main.rs (orden alfabético)
3. Si necesita estado compartido → añadir campo en AppState (api_server.rs)
4. Si expone herramientas → implementar trait Tool + registrar en main.rs
5. Si emite telemetría → clonar events_tx y usar events_tx.send(...)
6. Tests: módulo #[cfg(test)] al final del archivo
7. cargo test — verificar 0 regresiones, 0 warnings
```
