# SynaptoCore

**Motor Neurosimbólico de Conocimiento — 100% local, 0 dependencias de nube.**

SynaptoCore es un sistema de memoria semántica persistente con razonamiento autónomo. Almacena conocimiento en un grafo de tripletas (sujeto → relación → objeto), lo consolida en reposo mediante un bucle subconsciente, y lo expone a través de una API REST, una CLI y un dashboard en tiempo real. La inferencia LLM corre localmente via `llama.cpp` con aceleración NVIDIA RTX o CPU SIMD; sin llamadas a OpenAI, Anthropic ni ningún proveedor externo.

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Requisitos](#requisitos)
3. [Instalación y primer arranque](#instalación-y-primer-arranque)
4. [Configuración (.env)](#configuración-env)
5. [API REST](#api-rest)
6. [CLI — sc](#cli--sc)
7. [Memoria Federada (.synapse)](#memoria-federada-synapse)
8. [Features opcionales](#features-opcionales)
9. [Compilación de producción](#compilación-de-producción)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        SynaptoCore v0.17                         │
│                                                                   │
│  ┌──────────────┐   query   ┌──────────────────────────────────┐ │
│  │   API REST   │ ────────► │         LlmBridge / ReAct        │ │
│  │  (Axum 0.8)  │          │  correferencia → búsqueda →       │ │
│  │  port 8080   │ ◄──────── │  tool loop → síntesis            │ │
│  └──────┬───────┘  respuesta└────────────┬─────────────────────┘ │
│         │                               │                         │
│         │ learn                   read/write                      │
│         ▼                               ▼                         │
│  ┌──────────────┐          ┌────────────────────────┐            │
│  │  SessionMgr  │          │     MemoryGraph         │            │
│  │  (RAM/RwLock)│          │  SQLite WAL + embeddings│            │
│  │  ventana 6   │          │  tripletas + decay rate │            │
│  └──────────────┘          └────────────┬───────────┘            │
│                                          │                         │
│  ┌──────────────┐          ┌────────────▼───────────┐            │
│  │  ToolRegistry│          │    CuriosityEngine      │            │
│  │  TimeTool    │          │  (bucle subconsciente)  │            │
│  │  WebhookTool │          │  nodos → hipótesis →    │            │
│  │  Conectores  │          │  SearXNG → validación → │            │
│  │  (JSON)      │          │  consolidación en grafo  │            │
│  └──────────────┘          └─────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Módulos y su analogía neurológica

| Módulo | Archivo | Analogía | Función |
|--------|---------|----------|---------|
| **Memory Graph** | `memory_graph.rs` | Hipocampo | Grafo de tripletas semánticas en SQLite WAL. Embeddings f32 (BGE-Small). Similitud coseno. |
| **LLM Bridge** | `llm_bridge.rs` | Córtex Prefrontal | Extracción de entidades, síntesis de respuesta, bucle ReAct multi-paso. |
| **Session Manager** | `working_memory.rs` | Lóbulo Frontal | Memoria de trabajo en RAM. Ventana deslizante de 6 mensajes por sesión. Resolución de correferencia. |
| **Curiosity Engine** | `subconscious.rs` | Subconsciente / sueño REM | Bucle autónomo: muestrea pares de nodos, formula hipótesis, valida via SearXNG, consolida tripletas. |
| **Forgetting Curve** | `forgetting_curve.rs` | Olvido de Ebbinghaus | Decaimiento exponencial de aristas. Poda de conexiones débiles. Refuerzo Hebbiano en consulta. |
| **Truth Filter** | `truth_filter.rs` | Corteza Frontal (inhibición) | Validación epistémica 3-way (Verificado / Contradictorio / Inconcluso) antes de consolidar. |
| **Tool Registry** | `tools.rs` | Córtex Motor | Trait `Tool`. ReAct loop: decide herramienta → ejecuta → acumula en scratchpad → sintetiza. |
| **Federated Memory** | `federated_memory.rs` | Red Neuronal Distribuida | Intercambio de deltas entre agentes via archivos `.synapse`. |
| **API Server** | `api_server.rs` | Interfaz Sensoriomotora | Axum 0.8. REST + WebSocket de telemetría. |

### Flujo de una consulta

```
Usuario: "¿En qué lenguaje está hecho?"
(sesión activa: antes preguntó qué es PostgreSQL)

1. SessionManager.get_history()  → recupera los últimos mensajes
2. Planificador (1 llamada)      → intención, reescritura con el historial,
                                   entidades y herramienta: "¿En qué lenguaje está hecho PostgreSQL?"
3. Memoria                       → contexto semántico del grafo para cada entidad
4. Si la memoria no lo sabe      → busca en la web (SearXNG) y responde citando [n]
5. Respuesta en streaming        → y, si investigó, aprende lo verificado en segundo plano
6. SessionManager.push()         → guarda el turno
7. API → { response, session_id, answered_from, sources }
```

### Bucle Subconsciente (ReAct autónomo)

El `CuriosityEngine` corre en un hilo Tokio separado mientras el servidor está activo:

```
cada N minutos:
  1. Elige un par de conceptos relacionados y sin conectar (ej. "PostgreSQL", "MVCC")
  2. LLM formula una pregunta: "¿Cómo usa PostgreSQL el MVCC?"
  3. SearXNG busca evidencia en la web (si SEARXNG_URL está definida)
  4. EpistemicValidator: Verificado / Contradictorio / Inconcluso
  5. Si Verificado → insert_edge("postgresql", "usa", "mvcc", 0.8)
  6. ForgettingCurve.run_nightly_cleanup() → decae aristas sin refuerzo
```

---

## Requisitos

### Mínimos (CPU, sin LLM real)

- **Rust 1.82+** — `rustup update stable`
- **Windows 10/11 x64** (también compila en Linux/macOS)
- 4 GB RAM, cualquier CPU con SSE4.2

### Producción con GPU (recomendado)

- **NVIDIA RTX** (8 GB+ VRAM recomendado para modelos 7B Q4)
- **CUDA Toolkit 12.x** — [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- El modelo GGUF en disco (ver [Instalación](#instalación-y-primer-arranque))

### Opcionales

- **Ollama** — alternativa a llama.cpp para gestión de modelos
- **SearXNG** — instancia privada para el motor epistémico (TruthFilter + subconsciente)
- **curl** — usado por `start_brain.bat` para verificar que el servidor levantó

---

## Instalación y primer arranque

```powershell
# 1. Clonar el repositorio
git clone <repo-url> SynaptoCore
cd SynaptoCore

# 2. Configurar el entorno
Copy-Item .env.example .env
notepad .env     # ajusta las rutas y variables (ver sección siguiente)

# 3. Opción A — Un solo clic (Windows):
.\start_brain.bat              # CPU, ServerMock
.\start_brain.bat gpu          # RTX + llama.cpp
.\start_brain.bat gpu embed    # RTX + embeddings semánticos

# 3. Opción B — PowerShell avanzado:
.\start_brain.ps1 -Gpu -Embeddings

# 3. Opción C — Cargo directo:
cargo run --release                                    # CPU
cargo run --release --features cuda                    # RTX
cargo run --release --features "cuda,embeddings"       # RTX + semántica
```

El dashboard abre automáticamente en `http://localhost:8080/` al detectar que el servidor responde.

### Descargar un modelo GGUF

```powershell
# Opción A — HuggingFace CLI (instala con: pip install huggingface_hub[cli])
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF `
    --include "qwen2.5-7b-instruct-q4_k_m.gguf" `
    --local-dir C:\Models

# Opción B — Ollama (gestiona modelos automáticamente)
ollama pull qwen2.5:7b
# Luego en .env: apunta SYNAPTOCORE_MODEL_PATH al archivo en ~/.ollama/models/...

# Opción C — Modelo pequeño para pruebas (1.5 GB, 0.5B parámetros)
huggingface-cli download bartowski/Qwen2.5-0.5B-Instruct-GGUF `
    --include "qwen2.5-0.5b-instruct-q8_0.gguf" `
    --local-dir C:\Models
```

Modifica `SYNAPTOCORE_MODEL_PATH` en `.env` con la ruta al archivo descargado.

---

## Configuración (.env)

```bash
# ── LLM ────────────────────────────────────────────────────────────────────
SYNAPTOCORE_MODEL_PATH=C:\Models\qwen2.5-7b-instruct-q4_k_m.gguf
SYNAPTOCORE_GPU_LAYERS=999          # 999 = todas las capas en VRAM

# ── Base de Datos ───────────────────────────────────────────────────────────
SC_DB_PATH=synaptocore_memory.db    # ruta relativa al directorio del proyecto

# ── Servidor API ────────────────────────────────────────────────────────────
SC_HOST=127.0.0.1
SC_PORT=8080

# ── Motor Epistémico (opcional) ─────────────────────────────────────────────
SEARXNG_URL=http://localhost:8888   # sin esta var → subconsciente desconectado

# ── Conectores externos (opcional) ──────────────────────────────────────────
SYNAPTOCORE_CONNECTORS=conectores.json   # sin archivo → sin conectores
ERP_API_TOKEN=...                        # secretos citados en el JSON como ${ERP_API_TOKEN}

# ── Logging ─────────────────────────────────────────────────────────────────
RUST_LOG=synaptocore=info,tower_http=warn
```

Copia completa documentada en `.env.example`.

---

## Conectores externos

Cualquier sistema con API HTTP (ERP, CRM, inventario, una API propia, un motor de cálculo) se conecta
declarándolo en `conectores.json` (ruta configurable con `SYNAPTOCORE_CONNECTORS`), sin tocar el código.
El planificador ve el nombre y la descripción de cada conector y solo rellena sus parámetros: nunca
elige URLs. Los secretos van en variables de entorno y se citan como `${NOMBRE}`.

```json
[
  {
    "name": "Inventario",
    "description": "Consulta el stock disponible y el precio de un producto.",
    "method": "GET",
    "url": "http://localhost:9000/api/v1/inventory?producto={producto}",
    "headers": { "Authorization": "Bearer ${ERP_API_TOKEN}" },
    "params": { "producto": "Nombre o SKU del producto" }
  }
]
```

Plantilla completa en `conectores.example.json`. Con `POST`, los parámetros se envían como JSON. Un
conector mal declarado (URL con un `{parámetro}` sin describir, variable de entorno ausente) se omite al
arrancar con su motivo; los demás siguen funcionando.

## API REST

Base URL: `http://localhost:8080`

### `GET /health`

Estado del servidor y métricas del grafo de memoria.

```bash
curl http://localhost:8080/health
```
```json
{
  "status": "ok",
  "node_count": 47,
  "edge_count": 123,
  "avg_weight": 0.814,
  "version": "0.17.0"
}
```

### `POST /api/v1/learn`

Enseña texto libre al cerebro. El LLM extrae tripletas semánticas, las valida con TruthFilter y las consolida en el grafo.

```bash
curl -X POST http://localhost:8080/api/v1/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "PostgreSQL es una base de datos relacional escrita en C que usa MVCC."}'
```
```json
{
  "triples_extracted": 3,
  "nodes_created": 2,
  "edges_created": 3,
  "llm_attempts": 1
}
```

### `POST /api/v1/query`

Consulta con memoria de sesión. Si omites `session_id` se genera uno nuevo y se devuelve en la respuesta.

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "¿En qué lenguaje está hecho PostgreSQL?", "session_id": "mi-sesion-1"}'
```
```json
{
  "response": "PostgreSQL está escrito en C.",
  "session_id": "mi-sesion-1",
  "answered_from": "memory",
  "sources": []
}
```

Multi-turno: el sistema reescribe automáticamente las referencias ambiguas usando el historial de la sesión:

```bash
# Turno 2 — "tiene" se resuelve a "PostgreSQL" por correferencia
curl -X POST http://localhost:8080/api/v1/query \
  -d '{"prompt": "¿Tiene versión móvil?", "session_id": "mi-sesion-1"}'
```

### `GET /api/v1/graph/all`

Exporta el grafo completo (nodos + aristas) para visualización o depuración.

```bash
curl http://localhost:8080/api/v1/graph/all
```
```json
{
  "nodes": [
    {"id": "postgresql", "name": "PostgreSQL", "weight": 0.95, ...},
    ...
  ],
  "edges": [
    {"source": "postgresql", "relation": "escrito_en", "target": "c", "weight": 0.87},
    ...
  ]
}
```

### `GET /api/v1/graph/neighbors/{node_id}`

Vecinos directos de un nodo (útil para explorar el grafo localmente).

```bash
curl "http://localhost:8080/api/v1/graph/neighbors/fénix_flow"
```

### `POST /api/v1/federation/export`

Exporta un delta del grafo como archivo `.synapse` para sincronización multi-agente.

```bash
curl -X POST http://localhost:8080/api/v1/federation/export
```

### `GET /ws/subconscious`

WebSocket de telemetría en tiempo real. Emite eventos del bucle subconsciente:

```
ws://localhost:8080/ws/subconscious
```
```json
{"event_type": "sleep_cycle", "message": "Consolidé: fénix_flow→implementado_en→python (0.80)", "timestamp": 1729651234}
{"event_type": "decay",       "message": "Podé 3 aristas con peso < 0.15", "timestamp": 1729651290}
```

### `GET /`

Dashboard HTML con métricas en vivo, visualización del grafo y feed del subconsciente (actualización WebSocket automática).

---

## CLI — sc

La CLI `sc` es un cliente HTTP que se comunica con un servidor SynaptoCore en ejecución. Requiere que el servidor esté activo.

```bash
# Instalar el binario sc en PATH:
cargo install --path . --bin sc
```

### Comandos disponibles

```
USAGE:
    sc <COMMAND>

COMMANDS:
    query   Consulta al cerebro (con memoria de sesión opcional)
    learn   Enseña texto o archivo al cerebro
    export  Exporta el grafo como delta .synapse
    watch   Suscribe al feed de telemetría en tiempo real (WebSocket)
    health  Muestra el estado del servidor y métricas del grafo
    help    Muestra ayuda de cualquier comando
```

### Ejemplos de uso

```bash
# Consulta simple
sc query "¿Qué es el aprendizaje por refuerzo?"

# Consulta con sesión persistente
sc query --session mi-proyecto "¿Qué hace SynaptoCore?"
sc query --session mi-proyecto "¿Y en qué lenguaje está escrito?"

# Aprender desde texto
sc learn "El aprendizaje por refuerzo es una rama del ML donde un agente aprende por recompensa."

# Aprender desde archivo
sc learn --file informe_tecnico.txt

# Ver el grafo de memoria en JSON
sc export --out grafo_backup.synapse

# Monitoreo en tiempo real del subconsciente
sc watch

# Estado del servidor
sc health
```

### Variable de entorno del cliente

```bash
# Por defecto sc apunta a http://127.0.0.1:8080
# Para apuntar a un servidor remoto:
set SC_HOST=http://192.168.1.100:8080
sc health
```

---

## Memoria Federada (.synapse)

El módulo `federated_memory` permite que múltiples instancias de SynaptoCore intercambien conocimiento sin una base de datos centralizada. El protocolo es simple: cada agente exporta un **delta** (conjunto de operaciones desde el último sync) a un archivo `.synapse`, y cualquier otro agente puede importarlo.

### Formato del archivo .synapse

Un `.synapse` es JSON con la siguiente estructura:

```json
{
  "delta_id": "a3f2c1d4-...",
  "agent_id": "synaptocore-abc123",
  "timestamp": 1729651234,
  "operations": [
    {
      "type": "UpsertNode",
      "node": {
        "id": "postgresql",
        "name": "PostgreSQL",
        "weight": 0.95,
        "decay_rate": 0.01,
        "embedding": null
      }
    },
    {
      "type": "UpsertEdge",
      "source_id": "fénix_flow",
      "relation": "implementado_en",
      "target_id": "python",
      "weight": 0.87
    }
  ]
}
```

### Flujo de sincronización

```
Agente A                     Directorio compartido           Agente B
────────                     ─────────────────────           ────────
/api/v1/federation/export ──► delta_a1b2c3.synapse ──► POST /api/v1/federation/import
                                                           (merge con resolución
                                                            de conflictos por timestamp)
```

### Casos de uso

- **Multi-sucursal**: cada tienda tiene su instancia de SynaptoCore. Sincronizan knowledge via `.synapse` en un directorio compartido (NAS, S3, SharePoint).
- **Backup incremental**: exportar el grafo como delta y restaurarlo en otra máquina.
- **Especialización**: un agente aprende sobre inventario, otro sobre clientes. Ambos comparten el núcleo común via `.synapse`.

---

## Features opcionales

| Feature | Activación | Descripción |
|---------|-----------|-------------|
| `llama-backend` | `--features llama-backend` | Backend LLM real (CPU con SIMD). Requiere modelo GGUF. |
| `cuda` | `--features cuda` | llama.cpp con aceleración NVIDIA CUDA. Implica `llama-backend`. |
| `vulkan` | `--features vulkan` | llama.cpp con aceleración Vulkan (AMD, Intel Arc). Implica `llama-backend`. |
| `embeddings` | `--features embeddings` | fastembed (BGE-Small-EN v1.5). Habilita `semantic_search()`. |

Sin ningún feature, el servidor arranca con `ServerMock` — útil para CI, tests, y desarrollo del grafo de memoria sin necesidad del modelo LLM.

---

## Compilación de producción

```powershell
# Release optimizado (fat LTO, panic=abort, strip debuginfo):
cargo build --release

# Release con GPU RTX:
cargo build --release --features cuda

# Release completo (RTX + semántica):
cargo build --release --features "cuda,embeddings"

# Resultado: target/release/synaptocore.exe  (~4.7 MB)
#            target/release/sc.exe           (CLI)
```

El perfil `[profile.release]` en `Cargo.toml` aplica `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`, `panic = "abort"` y `strip = "debuginfo"`. La primera compilación release tarda 1-3 minutos; las incrementales ~30 segundos.

Para profiling con NVIDIA Nsight usa el perfil alternativo:

```powershell
cargo build --profile release-gpu --features cuda
# Binario en: target/release-gpu/synaptocore.exe (incluye símbolos de línea)
```

---

## Tests

```powershell
# Suite completa (242 tests, ~2.5s):
cargo test

# Un módulo específico:
cargo test connectors

# Con logs visibles:
cargo test -- --nocapture
```

Los tests de integración que requieren servicios externos (SearXNG) están marcados con `#[ignore]` y no corren por defecto.

---

## Licencia

MIT — ver `LICENSE`.
