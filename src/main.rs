// ============================================================
// main.rs — SynaptoCore v0.17
//   Paso 1:  Memory Graph (SQLite WAL)
//   Paso 2:  LLM Bridge (NeuralBridge + extracción + síntesis)
//   Paso 3:  Truth Filter (EpistemicValidator 3-way)
//   Paso 4:  Bucle Subconsciente (CuriosityEngine)
//   Paso 5:  Curva del Olvido (ForgettingCurve + refuerzo Hebbiano)
//   Paso 6:  Memoria Federada (FederationHub + delta-grafos)
//   Paso 7:  Córtex de Comunicación (Axum API + subconscious spawn)
//   Paso 8:  Panel de Observabilidad (Dashboard HTML embebido)
//   Paso 9:  Telemetría WebSocket (broadcast + /ws/subconscious)
//   Paso 10: Conexión Epistémica al Mundo Real (SearXNG HTTP real)
//   Paso 11: CLI Periférica (sc — SynaptoCore CLI)
//   Paso 12: Búsqueda Semántica Vectorial (fastembed + coseno)
//   Paso 13: Lóbulo Frontal (SessionManager + Correferencia)
//   Paso 14: Córtex Motor (Tool Use — ReAct pattern)
//   Paso 15: Orquestador Multi-Paso (ReAct Loop + Scratchpad)
//   Paso 16: Integración Empresarial (Fénix Flow Tools)
//   Paso 17: El Despertar (Optimización Release + Entorno)
//   Paso 21: Concurrencia Cognitiva (locks breves + prioridad del usuario)
// ============================================================
#![allow(dead_code)]

mod api_server;
mod error;
mod federated_memory;
mod fenix_tools;
mod forgetting_curve;
mod llm_bridge;
mod memory_graph;
mod ollama_backend;
mod priority;
mod prompts;
mod research;
mod subconscious;
mod tools;
mod truth_filter;
mod types;
mod working_memory;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use memory_graph::MemoryGraph;
use llm_bridge::{BridgeError, LlmBackend};
use ollama_backend::OllamaBackend;
use priority::{BackgroundLlm, CognitivePriority, YieldEvent};
use truth_filter::{BoxedSearch, MockSearchBackend, SearxngBackend};
use forgetting_curve::{ForgettingConfig, ForgettingCurve};
use subconscious::{CuriosityEngine, CuriosityError, CycleBackoff};
use api_server::{build_router, AppState, SubconsciousEvent};
use fenix_tools::{ClientTool, InventoryTool};
use tools::{TimeTool, ToolRegistry, WebhookTool};
use types::Node;
use working_memory::SessionManager;

// ── Mock LLM para el binario del servidor ────────────────────────────────────
//
// En producción: `cargo run --release --features cuda`
// activa LlamaCppBackend (cargado en init_brain).
//
// ServerMock simula respuestas razonables sin LLM real, detectando por
// contenido del prompt si se trata de extracción de entidades, tripletas
// o síntesis libre. Es deliberadamente simple para el demo.
struct ServerMock;

impl LlmBackend for ServerMock {
    fn complete(
        &self,
        system: &str,      // antes era _system (ignorado) — ahora se inspecciona
        user:   &str,
        _max:   i32,
        _temp:  f32,
    ) -> Result<String, BridgeError> {
        // Heurística de detección de tipo de prompt por palabras clave.
        // Inspeccionamos AMBOS prompts: el sistema define el "modo" de operación,
        // el usuario aporta los datos concretos.
        let u = user.to_lowercase();
        let s = system.to_lowercase();

        if s.contains("planificador de synaptocore") {
            // Paso 22: planificador. Heurística de intención; sin herramientas
            // ni reescritura (sin LLM real no hay correferencia que resolver).
            let message = user
                .lines()
                .find_map(|l| l.strip_prefix("Mensaje: "))
                .unwrap_or(user)
                .trim()
                .trim_matches('"');
            let m = message.to_lowercase();
            let learn = !message.starts_with('¿')
                && ["es un", "es una", "recuerda", "memoriza", "tiene"].iter().any(|k| m.contains(k));
            Ok(serde_json::json!({
                "i": if learn { "learn" } else { "query" },
                "q": message,
                "e": [],
                "t": "none",
                "p": {}
            }).to_string())
        } else if s.contains("síntesis de conocimiento") || s.contains("sintesis de conocimiento") {
            // Paso 4 — CuriosityEngine: síntesis de tripletas del bucle subconsciente.
            // El system prompt es curiosity_synthesis_system_prompt(node_a, node_b).
            // Extraemos los nodos del system prompt para generar una triple coherente.
            // Formato: "...conecten directamente «NODE_A» con «NODE_B»..."
            let node_a = s.split('«').nth(1)
                .and_then(|s| s.split('»').next())
                .unwrap_or("nodo_a");
            let node_b = s.split('«').nth(2)
                .and_then(|s| s.split('»').next())
                .unwrap_or("nodo_b");
            // Genera una relación plausible entre los dos nodos en snake_case.
            let a_snake = node_a.replace(' ', "_").replace('-', "_");
            let b_snake = node_b.replace(' ', "_").replace('-', "_");
            Ok(format!(
                r#"{{"triples": [{{"subject": "{a}", "relation": "relacionado_con", "object": "{b}", "weight": 0.75}}]}}"#,
                a = a_snake,
                b = b_snake,
            ))
        } else if s.contains("eres synaptocore, un asistente") {
            // Paso 24: respuestas en modo memoria, investigación, conocimiento
            // general o conversación. Sin LLM real: texto de demostración.
            Ok("SynaptoCore es una arquitectura cognitiva autónoma implementada en Rust. \
                Combina un grafo de conocimiento SQLite con inferencia LLM local y \
                validación epistémica de hechos.".to_string())
        } else if s.contains("extrae de las fuentes") {
            // Paso 24: aprendizaje a partir de una investigación.
            Ok(r#"{"triples": [{"subject": "synaptocore", "relation": "usa", "object": "rust", "weight": 0.9}]}"#.to_string())
        } else if u.contains("triples") || u.contains("tripleta") || u.contains("extract") {
            // Extracción de conocimiento para extract_and_memorize
            Ok(r#"{"triples": [
                {"subject": "synaptocore", "relation": "es", "object": "agente_cognitivo", "weight": 0.95},
                {"subject": "agente_cognitivo", "relation": "usa", "object": "rust", "weight": 0.90}
            ]}"#.to_string())
        } else if u.contains("question") || u.contains("pregunta") || u.contains("curiosidad") {
            // CuriosityEngine: generación de preguntas de investigación
            Ok(r#"{"question": "¿Cómo se relaciona SynaptoCore con Rust?", "reasoning": "Investigar vínculo técnico."}"#.to_string())
        } else if u.contains("verified") || u.contains("validar") || u.contains("verifid") {
            // TruthFilter: verificación de tripletas
            Ok(r#"{"status": "VERIFIED", "confidence": 0.88, "reasoning": "Demo: asumiendo verificado."}"#.to_string())
        } else {
            // Respuesta de síntesis libre (query_with_memory respuesta final)
            Ok("SynaptoCore es una arquitectura cognitiva autónoma implementada en Rust. \
                Combina un grafo de conocimiento SQLite con inferencia LLM local y \
                validación epistémica de hechos.".to_string())
        }
    }

    fn stream_synthesis(
        &self,
        system: &str,
        user:   &str,
        max:    i32,
        temp:   f32,
    ) -> Result<llm_bridge::BoxedTokenStream, BridgeError> {
        let full = self.complete(system, user, max, temp)?;
        let words: Vec<String> = full
            .split_inclusive(' ')
            .map(String::from)
            .collect();
        let stream = async_stream::stream! {
            for word in words {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                yield Ok(word);
            }
        };
        Ok(Box::pin(stream))
    }
}

// ServerMock es trivialmente Clone (sin estado mutable).
impl Clone for ServerMock {
    fn clone(&self) -> Self { ServerMock }
}

// ── Punto de entrada asíncrono ────────────────────────────────────────────────
//
// #[tokio::main] convierte la función en la entrada del runtime multi-hilo
// de Tokio. Con features = ["full"] en Cargo.toml, obtenemos:
//   - Thread pool para spawn_blocking (SQLite + LLM inference)
//   - Timer para sleep en el bucle subconsciente
//   - Signal handling para Ctrl+C graceful shutdown
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  SynaptoCore v0.17 — El Despertar (Producción Ready)              ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    // ── 1. Inicializar el grafo de memoria ────────────────────────────────────
    let graph = MemoryGraph::open("synaptocore_memory.db")?;
    println!("✓ Memory graph inicializado (synaptocore_memory.db)\n");

    // ── 1b. Memoria semántica (Paso 22) ──────────────────────────────────────
    // Carga del modelo (y descarga la primera vez) en spawn_blocking: es E/S
    // y CPU bloqueante. Si falla (sin red en el primer arranque, sin la
    // feature), el sistema sigue en modo léxico en lugar de no arrancar.
    let graph = Arc::new(Mutex::new(attach_embeddings(graph).await));

    // Backfill en segundo plano de los nodos sin vector (anteriores al modelo
    // o cuyo embedding falló). Por lotes y sin retener el lock al vectorizar.
    let graph_bf = Arc::clone(&graph);
    tokio::spawn(async move {
        match tokio::task::spawn_blocking(move || memory_graph::backfill_embeddings(&*graph_bf, 64)).await {
            Ok(Ok(n)) if n > 0 => println!("✓ Backfill semántico: {n} nodos vectorizados"),
            Ok(Err(e)) => eprintln!("⚠ Backfill semántico interrumpido: {e}"),
            _ => {}
        }
    });

    // ── 2. Poblar el grafo con conocimiento inicial (idempotente) ─────────────
    {
        let g = graph.lock().unwrap();
        for (id, name) in &[
            ("synaptocore",   "SynaptoCore"),
            ("rust",          "Rust"),
            ("sqlite",        "SQLite"),
            ("axum",          "Axum"),
            ("conocimiento",  "Conocimiento"),
        ] {
            let _ = g.insert_node(&Node::new(*id, "concept", *name));
        }
        let _ = g.insert_edge("synaptocore", "implementado_en", "rust",         0.99);
        let _ = g.insert_edge("synaptocore", "usa_como_db",     "sqlite",       0.98);
        let _ = g.insert_edge("synaptocore", "expone_api_con",  "axum",         0.95);
        let _ = g.insert_edge("synaptocore", "gestiona",        "conocimiento", 0.97);
        println!("✓ Conocimiento base insertado en el grafo\n");
    }

    // ── 3. Inicializar el backend LLM ─────────────────────────────────────────
    // Prioridad: OllamaBackend (si Ollama responde y el modelo existe) →
    // ServerMock (fallback demo sin LLM real).
    // Para llama.cpp directo: cargo run --features cuda (carga GGUF local).
    //
    // spawn_blocking (Paso 21): OllamaBackend usa `reqwest::blocking`, que no
    // puede construirse ni usarse en un hilo del runtime async. En debug,
    // reqwest lo comprueba creando y destruyendo un runtime interno, lo que
    // provocaba el panic "Cannot drop a runtime in a context where blocking is
    // not allowed" al arrancar. En release esa comprobación no existe, pero el
    // health-check seguía bloqueando un worker de Tokio.
    let (llm, llm_label) = tokio::task::spawn_blocking(
        || -> (Arc<dyn LlmBackend + Send + Sync + 'static>, String) {
            let ollama = OllamaBackend::from_env();
            let label_candidate = format!("Ollama ({})", ollama.model);
            match ollama.health_check() {
                Ok(()) => {
                    println!("✓ Backend LLM: {}\n", label_candidate);
                    (Arc::new(ollama), label_candidate)
                }
                Err(e) => {
                    println!("⚠ Ollama no disponible: {e}");
                    println!("  Usando ServerMock (demo sin LLM real)\n");
                    (Arc::new(ServerMock), "ServerMock (fallback)".to_string())
                }
            }
        },
    ).await?;

    // Canal de broadcast para telemetría en tiempo real (Paso 9).
    // Capacidad 128: si el cliente WS es lento y el buffer se llena,
    // los mensajes más antiguos se descartan (preferimos datos frescos).
    // `broadcast::Sender` es Clone: se comparte con AppState y con el loop
    // del subconsciente sin necesidad de Arc adicional.
    let (events_tx, _events_rx_dummy) = tokio::sync::broadcast::channel::<String>(128);

    // Paso 14: registro de herramientas para el ciclo ReAct.
    // Paso 16: herramientas empresariales Fénix Flow (InventoryTool, ClientTool).
    // TimeTool:      fecha/hora del sistema (sin deps externos).
    // WebhookTool:   GET HTTP a sistemas externos (Fénix Flow, ERPs, APIs).
    // InventoryTool: consulta de stock y precios en el ERP Fénix Flow.
    // ClientTool:    historial y datos de clientes en el ERP Fénix Flow.
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(TimeTool);
    tool_registry.register(WebhookTool);
    tool_registry.register(InventoryTool);
    tool_registry.register(ClientTool);
    let tool_registry = Arc::new(tool_registry);

    // Paso 21: árbitro entre el usuario (primer plano) y el subconsciente (fondo).
    let priority = Arc::new(CognitivePriority::default());

    // Paso 10: backend de búsqueda. Si SEARXNG_URL está definida, el cerebro
    // valida hechos contra internet. Se construye una sola vez y fuera del
    // runtime async, por el mismo motivo que OllamaBackend.
    // Paso 24: el chat también lo usa para investigar lo que no sabe.
    let search: Option<BoxedSearch> = tokio::task::spawn_blocking(|| {
        std::env::var("SEARXNG_URL").ok().map(|url| {
            println!("✓ Investigación web REAL (chat y subconsciente): SearXNG en {url}\n");
            BoxedSearch(Arc::new(SearxngBackend::new(url)))
        })
    }).await?;
    if search.is_none() {
        println!("ℹ Sin SEARXNG_URL: lo que no esté en memoria se responde con conocimiento general (no verificado)\n");
    }

    let state = AppState {
        graph:         Arc::clone(&graph),
        llm:           Arc::clone(&llm),
        agent_id:      format!("synaptocore-{}", uuid::Uuid::new_v4()),
        events_tx:     events_tx.clone(),
        sessions:      Arc::new(SessionManager::new()),
        tool_registry: Arc::clone(&tool_registry),
        priority:      Arc::clone(&priority),
        search:        search.clone(),
    };

    // ── 4. Bucle Subconsciente en background (Paso 4 + Paso 9 + Paso 21) ────
    //
    // Patrón de concurrencia:
    //   • tokio::spawn → tarea asíncrona liviana (no bloquea el runtime)
    //   • tokio::task::spawn_blocking → CuriosityEngine es síncrono;
    //     spawn_blocking lo ejecuta en el thread-pool de Tokio para
    //     operaciones CPU/IO-bound sin bloquear los hilos async.
    //   • tokio::time::sleep → espera cooperativa entre ciclos (no bloquea).
    //   • Paso 21: el motor recibe `&Mutex<MemoryGraph>` y solo lo bloquea al
    //     leer el vacío y al consolidar, nunca durante inferencia o búsqueda.
    //   • Paso 21: el LLM va envuelto en `BackgroundLlm`, que cede el turno al
    //     chat antes de cada inferencia; tampoco se arranca un ciclo con el
    //     usuario activo.
    //
    // El loop se cancela automáticamente cuando main() termina (Ctrl+C).

    // Sin SearXNG el subconsciente valida sin fuentes (siempre INCONCLUSIVE).
    let search_sc: BoxedSearch = search
        .clone()
        .unwrap_or_else(|| BoxedSearch(Arc::new(MockSearchBackend::empty())));

    let tx_yield = events_tx.clone();
    let llm_sc: Arc<dyn LlmBackend + Send + Sync + 'static> = Arc::new(
        BackgroundLlm::new(Arc::clone(&llm), Arc::clone(&priority)).on_yield(move |e| {
            let msg = match e {
                YieldEvent::Paused =>
                    "⏸ Pausa cognitiva: el chat tiene prioridad sobre el subconsciente".to_string(),
                YieldEvent::Resumed(d) =>
                    format!("▶ Reanudado tras {:.1} s cediendo el turno al chat", d.as_secs_f64()),
            };
            println!("  [🧠 subconscious] {msg}");
            let _ = tx_yield.send(SubconsciousEvent::new("Cycle", msg).to_json());
        }),
    );
    let graph_sc    = Arc::clone(&graph);
    let events_sc   = events_tx.clone();   // Paso 9: canal para telemetría WS
    let priority_sc = Arc::clone(&priority);

    tokio::spawn(async move {
        println!(
            "[subconsciente] Motor de curiosidad continuo iniciado (ciclo cada {}–{} s según lo que aprenda)\n",
            subconscious::DEFAULT_CURIOSITY_INTERVAL_SECS,
            subconscious::MAX_CURIOSITY_INTERVAL_SECS,
        );
        // Paso 23: el intervalo crece mientras los ciclos no aprendan nada.
        let mut backoff = CycleBackoff::standard();
        loop {
            // Paso 21: no arrancar un ciclo mientras el usuario está activo.
            priority_sc.wait_idle().await;

            let graph_ref  = Arc::clone(&graph_sc);
            let llm_ref    = Arc::clone(&llm_sc);
            let search     = search_sc.clone();
            let tx         = events_sc.clone();

            // spawn_blocking: CuriosityEngine::run_sleep_cycle es síncrono.
            // Devuelve si el ciclo consolidó conocimiento nuevo.
            let productive = tokio::task::spawn_blocking(move || {
                // Closure de logging que emite SubconsciousEvent al canal WS.
                // Si no hay clientes conectados, send() devuelve SendError
                // que se descarta con let _ = ... (sin panic, Paso 9).
                let tx_log = tx.clone();
                let log_fn = move |msg: &str| {
                    // Clasificar el evento por palabras clave del mensaje
                    // para asignar el event_type correcto al frontend JS.
                    let event_type = if msg.contains("Consolidada") || msg.contains("Aprendido") {
                        "Consolidation"
                    } else if msg.contains("Validación") || msg.contains("Validacion") {
                        "Validation"
                    } else if msg.contains("Pregunta generada") {
                        "Hypothesis"
                    } else if msg.contains("descartada") || msg.contains("❌") {
                        "Rejection"
                    } else if msg.contains("Error") || msg.contains("⚠") {
                        "Error"
                    } else {
                        "Cycle"
                    };
                    let event = SubconsciousEvent::new(event_type, msg);
                    // Ignorar SendError: normal si no hay clientes WS.
                    let _ = tx_log.send(event.to_json());
                    // También imprimir en stdout para logs del servidor.
                    println!("  [🧠 subconscious] {}", msg);
                };

                // La type-erasure via BoxedSearch permite que CuriosityEngine<L, BoxedSearch>
                // sea el mismo tipo monomorphizado independientemente del backend elegido.
                let engine = CuriosityEngine::with_config(llm_ref, search, 0.5, log_fn);

                // Se pasa el Mutex, no un guard: el lock se toma por fases (Paso 21).
                let reports = engine.run_sleep_cycle(&*graph_ref, 1);
                let mut productive = false;
                for r in reports {
                    match r {
                        Ok(report) if !report.triples_consolidated.is_empty() => {
                            productive = true;
                            let msg = format!(
                                "✓ Aprendido: {}",
                                report.triples_consolidated.join(", ")
                            );
                            println!("[subconsciente] {}", msg);
                            // Evento de consolidación final con lista de tripletas.
                            let _ = tx.send(
                                SubconsciousEvent::new("Consolidation", msg).to_json()
                            );
                        }
                        Ok(_)  => {} // Ciclo sin nuevas tripletas; silencioso.
                        // Paso 23: sin vacíos elegibles el motor descansa; no es un error.
                        Err(CuriosityError::NoEligibleGap) => {
                            let msg = format!("💤 {}", CuriosityError::NoEligibleGap);
                            println!("  [🧠 subconscious] {msg}");
                            let _ = tx.send(SubconsciousEvent::new("Cycle", msg).to_json());
                        }
                        Err(e) => {
                            let msg = format!("⚠ Error de ciclo: {}", e);
                            eprintln!("[subconsciente] {}", msg);
                            let _ = tx.send(
                                SubconsciousEvent::new("Error", msg).to_json()
                            );
                        }
                    }
                }
                productive
            }).await.unwrap_or(false);

            // Paso 23: espera adaptativa, troceada en pasos del intervalo base
            // para despertar en cuanto el grafo cambie por otra vía (el
            // usuario enseña algo, la federación importa nodos).
            backoff.record(productive);
            if backoff.current() > backoff.base() {
                let msg = format!(
                    "💤 Sin conocimiento nuevo; próximo ciclo en {} s",
                    backoff.current().as_secs()
                );
                println!("  [🧠 subconscious] {msg}");
                let _ = events_sc.send(SubconsciousEvent::new("Cycle", msg).to_json());
            }
            let before = graph_signature(&graph_sc).await;
            let deadline = tokio::time::Instant::now() + backoff.current();
            while tokio::time::Instant::now() < deadline {
                let step = backoff.base().min(deadline - tokio::time::Instant::now());
                tokio::time::sleep(step).await;
                if graph_signature(&graph_sc).await != before {
                    backoff.reset();
                    break;
                }
            }
        }
    });

    // ── 5. Limpieza nocturna de memoria (ForgettingCurve) ─────────────────────
    //
    // Corre una vez al iniciar para limpiar aristas débiles del disco,
    // y luego periódicamente cada hora en background.
    let graph_fc = Arc::clone(&graph);
    tokio::spawn(async move {
        loop {
            let graph_ref = Arc::clone(&graph_fc);
            tokio::task::spawn_blocking(move || {
                let Ok(graph) = graph_ref.lock() else { return; };
                let fc = ForgettingCurve::new(ForgettingConfig {
                    decay_rate:      0.00001, // λ: decae lentamente (~3 días a w=1.0)
                    prune_threshold: 0.05,    // θ: poda aristas con peso < 5%
                });
                match fc.run_nightly_cleanup(&graph) {
                    Ok(r) if r.edges_pruned > 0 =>
                        println!("[forgetting] Podadas {} aristas · {} nodos huérfanos",
                            r.edges_pruned, r.nodes_removed),
                    Ok(_)  => {} // Nada que podar.
                    Err(e) => eprintln!("[forgetting] Error en limpieza: {}", e),
                }
            }).await.ok();

            // Ciclo de limpieza cada hora (3600 segundos).
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    });

    // ── 6. Iniciar el servidor Axum ───────────────────────────────────────────
    // SYNAPTOCORE_PORT permite levantar una segunda instancia (p. ej. sobre
    // una copia de la base) sin detener la principal.
    let port: u16 = std::env::var("SYNAPTOCORE_PORT").ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let router = build_router(state);

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  🧠 SynaptoCore API escuchando en {:<33}║", format!("http://{addr}"));
    println!("║                                                                    ║");
    println!("║  Endpoints disponibles:                                            ║");
    println!("║    GET  /health                     → Estado del sistema          ║");
    println!("║    POST /api/v1/query               → Consulta con memoria         ║");
    println!("║    POST /api/v1/learn               → Ingesta de conocimiento      ║");
    println!("║    GET  /api/v1/graph/neighbors/:id → Vecindario de un nodo        ║");
    println!("║    POST /api/v1/federation/export   → Delta incremental (24h)      ║");
    println!("║                                                                    ║");
    println!("║  Backend LLM: {:<52} ║", llm_label);
    println!("║  Bucle subconsciente: cada 3–60 s según lo que aprenda            ║");
    println!("║  Presiona Ctrl+C para detener.                                    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    // `axum::serve` bloquea hasta Ctrl+C. El select! con signal::ctrl_c()
    // permite un graceful shutdown sin perder datos del SQLite (WAL commit).
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    println!("\n✓ SynaptoCore detenido limpiamente. Base de datos guardada.");
    Ok(())
}

/// Conecta el motor de embeddings al grafo (Paso 22) y carga el índice
/// vectorial en RAM. Todo en `spawn_blocking`: cargar el modelo ONNX y leer
/// los BLOBs es bloqueante.
#[cfg(feature = "embeddings")]
async fn attach_embeddings(graph: MemoryGraph) -> MemoryGraph {
    let loaded = tokio::task::spawn_blocking(move || {
        match memory_graph::FastEmbedBackend::new() {
            Ok(backend) => Ok(graph.with_embedding(Arc::new(backend))),
            Err(e) => Err((graph, e)),
        }
    }).await;
    match loaded {
        Ok(Ok(graph)) => {
            println!(
                "✓ Memoria semántica: fastembed multilingüe ({} vectores en RAM)\n",
                graph.vector_index_len()
            );
            graph
        }
        Ok(Err((graph, e))) => {
            println!("⚠ Modelo de embeddings no disponible: {e}");
            println!("  Recuperación de contexto en modo léxico (LIKE)\n");
            graph
        }
        Err(e) => panic!("La carga del modelo de embeddings abortó: {e}"),
    }
}

#[cfg(not(feature = "embeddings"))]
async fn attach_embeddings(graph: MemoryGraph) -> MemoryGraph {
    println!("ℹ Compilado sin la feature 'embeddings': recuperación léxica (LIKE)\n");
    graph
}

/// Huella barata del grafo (nodos, aristas) para que el bucle subconsciente
/// detecte cambios hechos por otros mientras espera (Paso 23).
async fn graph_signature(graph: &Arc<Mutex<MemoryGraph>>) -> Option<(u64, u64)> {
    let graph = Arc::clone(graph);
    tokio::task::spawn_blocking(move || {
        let stats = graph.lock().ok()?.stats().ok()?;
        Some((stats.node_count, stats.edge_count))
    }).await.ok().flatten()
}

/// Espera la señal Ctrl+C para shutdown graceful.
///
/// Axum `with_graceful_shutdown` espera a que el Future resuelva antes
/// de cerrar el servidor, permitiendo que los handlers en vuelo terminen.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Fallo al instalar el manejador de Ctrl+C");
    println!("\n[signal] Ctrl+C recibido. Iniciando shutdown graceful...");
}

// ═════════════════════════════════════════════════════════════════════════════
//  TESTS DE INTEGRACIÓN — Pasos 1-7
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod integration_tests {
    use super::*;
    use llm_bridge::mock::MockLlmBackend;
    use llm_bridge::NeuralBridge;
    use truth_filter::{EpistemicValidator, MockSearchBackend, ValidationStatus};
    use subconscious::CuriosityEngine;
    use forgetting_curve::{ForgettingConfig, ForgettingCurve};
    use federated_memory::{FederationHub, MemoryDelta, DeltaOperation};
    use memory_graph::TraversalDirection;
    use types::{Node, Edge};
    use tempfile::TempDir;

    fn test_graph() -> MemoryGraph {
        MemoryGraph::open(":memory:").unwrap()
    }

    // ── Pipeline completo Pasos 1 + 2 + 3 ────────────────────────────────────
    #[test]
    fn test_full_three_stage_pipeline() {
        let graph = test_graph();

        let bridge = NeuralBridge::new(MockLlmBackend::new(vec![
            r#"{"triples": [
                {"subject": "sol",    "relation": "es_una",  "object": "estrella", "weight": 0.99},
                {"subject": "tierra", "relation": "orbita",  "object": "sol",      "weight": 0.99},
                {"subject": "luna",   "relation": "orbita",  "object": "tierra",   "weight": 0.98}
            ]}"#,
        ]));
        let extraction = bridge
            .extract_and_memorize(&graph, "El sol es una estrella. La Tierra orbita al Sol.")
            .unwrap();
        assert_eq!(extraction.triples.len(), 3);

        let validator = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"VERIFIED","confidence":0.99,"reasoning":"Hecho astronómico."}"#,
            ]),
            MockSearchBackend::new(vec![vec![
                "La Tierra orbita alrededor del Sol en aproximadamente 365 días.",
            ]]),
        );
        let r = validator.validate_triplet("tierra", "orbita", "sol", 0.99).unwrap();
        assert_eq!(r.status, ValidationStatus::Verified);
        assert!(r.should_accept());

        let validator_falso = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"CONTRADICTED","confidence":0.99,"reasoning":"El Sol no orbita la Tierra."}"#,
            ]),
            MockSearchBackend::new(vec![vec![
                "El heliocentrismo establece que la Tierra orbita al Sol.",
            ]]),
        );
        let r_falso = validator_falso
            .validate_triplet("sol", "orbita", "tierra", 0.85)
            .unwrap();
        assert_eq!(r_falso.status, ValidationStatus::Contradicted);
        assert!(!r_falso.should_accept());
    }

    #[test]
    fn test_contradicted_triplet_does_not_pollute_graph() {
        let graph = test_graph();
        graph.insert_node(&Node::new("tierra", "concept", "Tierra")).unwrap();
        graph.insert_node(&Node::new("sol",    "concept", "Sol")).unwrap();

        let validator = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"CONTRADICTED","confidence":0.98,"reasoning":"Falso."}"#,
            ]),
            MockSearchBackend::new(vec![vec!["La Tierra orbita al Sol."]]),
        );
        let result = validator.validate_triplet("sol", "orbita", "tierra", 0.90).unwrap();
        assert!(!result.should_accept());

        let hood = graph.get_neighbors("sol", TraversalDirection::Outgoing).unwrap();
        assert!(hood.neighbors.is_empty(), "Triplet CONTRADICTED no debe llegar al grafo");
    }

    #[test]
    fn test_verified_weight_is_adjusted_before_insertion() {
        let graph = test_graph();

        let validator = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"VERIFIED","confidence":0.80,"reasoning":"Confirmado."}"#,
            ]),
            MockSearchBackend::new(vec![vec!["fuente de confirmación"]]),
        );
        let result = validator.validate_triplet("a", "b", "c", 1.0).unwrap();
        assert!((result.adjusted_weight - 0.80).abs() < 1e-9);

        graph.insert_node(&Node::new("a", "concept", "A")).unwrap();
        graph.insert_node(&Node::new("c", "concept", "C")).unwrap();
        graph.insert_edge("a", "b", "c", result.adjusted_weight).unwrap();

        let hood = graph.get_neighbors("a", TraversalDirection::Outgoing).unwrap();
        assert!((hood.neighbors[0].0.weight - 0.80).abs() < 1e-9);
    }

    #[test]
    fn test_custom_threshold_controls_inconclusive() {
        let v_strict = EpistemicValidator::with_threshold(
            MockLlmBackend::new(vec![
                r#"{"status":"INCONCLUSIVE","confidence":0.6,"reasoning":"Poca evidencia."}"#,
            ]),
            MockSearchBackend::empty(),
            0.8,
        );
        let r1 = v_strict.validate_triplet("a", "rel", "b", 0.7).unwrap();
        assert!(!r1.should_accept(), "Umbral estricto debe rechazar confianza=0.6");

        let v_permissive = EpistemicValidator::with_threshold(
            MockLlmBackend::new(vec![
                r#"{"status":"INCONCLUSIVE","confidence":0.6,"reasoning":"Poca evidencia."}"#,
            ]),
            MockSearchBackend::empty(),
            0.4,
        );
        let r2 = v_permissive.validate_triplet("a", "rel", "b", 0.7).unwrap();
        assert!(r2.should_accept(), "Umbral permisivo debe aceptar confianza=0.6");
    }

    #[test]
    fn test_validation_pipeline_latency() {
        let validator = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"VERIFIED","confidence":0.9,"reasoning":"OK."}"#,
            ]),
            MockSearchBackend::new(vec![vec!["snippet rápido"]]),
        );

        let start   = std::time::Instant::now();
        let result  = validator.validate_triplet("a", "rel", "b", 0.8).unwrap();
        let elapsed = start.elapsed();

        assert!(result.should_accept());
        assert!(
            elapsed.as_millis() < 10,
            "Validación (mock) tardó {}ms, esperado < 10ms", elapsed.as_millis()
        );
    }

    // ── Tests de integración Paso 4: Bucle Subconsciente ─────────────────────

    #[test]
    fn test_subconscious_integration_grows_graph() {
        let graph = test_graph();
        graph.insert_node(&Node::new("rust",    "concept", "Rust")).unwrap();
        graph.insert_node(&Node::new("ml",      "concept", "ML")).unwrap();
        graph.insert_node(&Node::new("memoria", "concept", "Memoria")).unwrap();
        // El muestreo es aleatorio (Paso 23): rust ↔ ml es el único vacío.
        graph.insert_edge("rust", "usa", "memoria", 0.9).unwrap();
        graph.insert_edge("ml", "usa", "memoria", 0.9).unwrap();

        let engine = CuriosityEngine::new_silent(
            MockLlmBackend::new(vec![
                r#"{"question":"¿Puede Rust ejecutar ML?","reasoning":"Interop técnica."}"#,
                r#"{"status":"VERIFIED","confidence":0.90,"reasoning":"Confirmado."}"#,
                r#"{"triples":[{"subject":"rust","relation":"ejecuta","object":"ml","weight":0.88}]}"#,
            ]),
            MockSearchBackend::new(vec![
                vec!["Rust tiene crates para ML como burn y linfa."],
            ]),
        );

        let edges_before = graph.stats().unwrap().edge_count;
        let results = engine.run_sleep_cycle(&graph, 1);
        let report = results.into_iter().next().unwrap().unwrap();

        assert!(matches!(report.validation_status, ValidationStatus::Verified));
        let edges_after = graph.stats().unwrap().edge_count;
        assert!(edges_after > edges_before, "El grafo debe crecer: {} → {}", edges_before, edges_after);
    }

    #[test]
    fn test_subconscious_integration_rejects_false_hypothesis() {
        let graph = test_graph();
        graph.insert_node(&Node::new("sol",    "concept", "Sol")).unwrap();
        graph.insert_node(&Node::new("tierra", "concept", "Tierra")).unwrap();
        graph.insert_node(&Node::new("luna",   "concept", "Luna")).unwrap();

        let engine = CuriosityEngine::new_silent(
            MockLlmBackend::new(vec![
                r#"{"question":"¿Orbita el Sol alrededor de la Tierra?","reasoning":"Pregunta incorrecta."}"#,
                r#"{"status":"CONTRADICTED","confidence":0.99,"reasoning":"El heliocentrismo refuta esto."}"#,
            ]),
            MockSearchBackend::new(vec![
                vec!["La Tierra orbita al Sol, no al revés."],
            ]),
        );

        let edges_before = graph.stats().unwrap().edge_count;
        let results = engine.run_sleep_cycle(&graph, 1);
        let report = results.into_iter().next().unwrap().unwrap();

        assert!(matches!(report.validation_status, ValidationStatus::Contradicted));
        assert!(report.rejection_reason.is_some());
        let edges_after = graph.stats().unwrap().edge_count;
        assert_eq!(edges_before, edges_after, "Grafo contaminado por hipótesis rechazada");
    }

    // ── Tests de integración Paso 5: Curva del Olvido ────────────────────────

    #[test]
    fn test_reinforced_edge_survives_nightly_cleanup() {
        let graph = test_graph();
        graph.insert_node(&Node::new("a",      "c", "A")).unwrap();
        graph.insert_node(&Node::new("b",      "c", "B")).unwrap();
        graph.insert_node(&Node::new("c_node", "c", "C")).unwrap();

        let now  = types::current_unix_ts();
        let past = now - 100_000;

        graph.insert_edge("a", "rel1", "b",      0.5).unwrap();
        graph.insert_edge("a", "rel2", "c_node", 0.5).unwrap();

        graph.conn_execute_for_test(
            "UPDATE edges SET last_accessed = ?1 WHERE relation = 'rel1'",
            rusqlite::params![past],
        ).unwrap();

        graph.reinforce_edge("a", "rel2", "c_node").unwrap();

        let report = graph.apply_decay_and_prune(1.0, 0.10).unwrap();
        assert!(report.edges_pruned >= 1, "La arista antigua debe podarse");

        let hood = graph.get_neighbors("a", TraversalDirection::Outgoing).unwrap();
        assert!(
            hood.neighbors.iter().any(|(e, _)| e.relation == "rel2"),
            "La arista reforzada debe sobrevivir"
        );
    }

    #[test]
    fn test_full_five_stage_pipeline() {
        let graph = test_graph();

        graph.insert_node(&Node::new("cortisol",  "concept", "Cortisol")).unwrap();
        graph.insert_node(&Node::new("glucosa",   "concept", "Glucosa")).unwrap();
        graph.insert_node(&Node::new("obsoleto",  "concept", "Obsoleto")).unwrap();
        graph.insert_node(&Node::new("obsoleto2", "concept", "Obsoleto2")).unwrap();
        graph.insert_edge("cortisol", "eleva",    "glucosa",   0.85).unwrap();
        graph.insert_edge("obsoleto", "link_old", "obsoleto2", 0.20).unwrap();

        let past = types::current_unix_ts() - 1_000_000;
        graph.conn_execute_for_test(
            "UPDATE edges SET last_accessed = ?1 WHERE source_id = 'obsoleto'",
            rusqlite::params![past],
        ).unwrap();

        graph.reinforce_edge("cortisol", "eleva", "glucosa").unwrap();

        let fc = ForgettingCurve::new(ForgettingConfig { decay_rate: 0.00001, prune_threshold: 0.10 });
        let report = fc.run_nightly_cleanup(&graph).unwrap();

        let hood = graph.get_neighbors("cortisol", TraversalDirection::Outgoing).unwrap();
        assert!(!hood.neighbors.is_empty(), "cortisol→glucosa debe sobrevivir");
        assert!(hood.neighbors[0].0.weight >= 0.85);
        assert!(report.edges_pruned >= 1, "La arista obsoleta debe podarse");
    }

    // ── Tests de integración Paso 6: Memoria Federada ────────────────────────

    #[test]
    fn test_federation_snapshot_round_trip() {
        let dir   = TempDir::new().unwrap();
        let graph = test_graph();

        graph.insert_node(&Node::new("n1", "c", "N1")).unwrap();
        graph.insert_node(&Node::new("n2", "c", "N2")).unwrap();
        graph.insert_edge("n1", "rel", "n2", 0.70).unwrap();

        let hub_a = FederationHub::new("agent-a", dir.path()).unwrap();
        let hub_b = FederationHub::new("agent-b", dir.path()).unwrap();

        hub_a.export_full_snapshot(&graph).unwrap();

        let graph_b = MemoryGraph::open(":memory:").unwrap();
        let (report, errors) = hub_b.import_all_deltas(&graph_b).unwrap();

        assert!(errors.is_empty());
        assert_eq!(report.nodes_inserted, 2);
        assert_eq!(report.edges_inserted, 1);
    }

    #[test]
    fn test_federation_own_deltas_ignored() {
        let dir   = TempDir::new().unwrap();
        let graph = test_graph();
        let hub   = FederationHub::new("self-agent", dir.path()).unwrap();

        let delta = MemoryDelta::new("self-agent")
            .push(DeltaOperation::UpsertNode { node: Node::new("mine", "c", "Mine") });
        hub.export_delta(&delta).unwrap();

        let (report, _) = hub.import_all_deltas(&graph).unwrap();
        assert_eq!(report.nodes_inserted, 0, "Los deltas propios no se reimportan");
    }

    #[test]
    fn test_federation_collective_reinforcement() {
        let dir      = TempDir::new().unwrap();
        let graph    = test_graph();
        let hub_me   = FederationHub::new("me",   dir.path()).unwrap();
        let hub_peer = FederationHub::new("peer", dir.path()).unwrap();

        graph.insert_node(&Node::new("a", "c", "A")).unwrap();
        graph.insert_node(&Node::new("b", "c", "B")).unwrap();
        graph.insert_edge("a", "knows", "b", 0.50).unwrap();

        let edge  = Edge::new("e-peer", "a", "knows", "b", 0.50).unwrap();
        let delta = MemoryDelta::new("peer")
            .push(DeltaOperation::UpsertEdge { edge });
        hub_peer.export_delta(&delta).unwrap();

        let (report, _) = hub_me.import_all_deltas(&graph).unwrap();
        assert_eq!(report.edges_reinforced, 1);

        let hood = graph.get_neighbors("a", TraversalDirection::Outgoing).unwrap();
        assert!(
            hood.neighbors[0].0.weight > 0.50,
            "El refuerzo federado debe subir el peso"
        );
    }

    #[test]
    fn test_federation_delete_propagates() {
        let dir   = TempDir::new().unwrap();
        let graph = test_graph();
        let hub_a = FederationHub::new("a", dir.path()).unwrap();
        let hub_b = FederationHub::new("b", dir.path()).unwrap();

        graph.insert_node(&Node::new("old", "c", "Old")).unwrap();

        let delta = MemoryDelta::new("a")
            .push(DeltaOperation::DeleteNode { node_id: "old".to_string() });
        hub_a.export_delta(&delta).unwrap();

        let (report, _) = hub_b.import_all_deltas(&graph).unwrap();
        assert_eq!(report.nodes_deleted, 1);

        let results = graph.search_nodes("Old", 10).unwrap();
        assert!(results.is_empty(), "El nodo debe haberse eliminado por la federación");
    }

    // ── Tests de integración Paso 7: API Server ───────────────────────────────

    /// AppState puede crearse y su grafo mutearse desde tests síncronos.
    #[test]
    fn test_app_state_graph_is_shared_correctly() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let llm: Arc<dyn LlmBackend + Send + Sync + 'static> =
            Arc::new(MockLlmBackend::new(vec![] as Vec<String>));

        let state = AppState {
            graph:         Arc::clone(&graph),
            llm:           Arc::clone(&llm),
            agent_id:      "test-state".to_string(),
            events_tx:     tokio::sync::broadcast::channel::<String>(1).0,
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };

        // Insertar desde el Arc original
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("test_node", "c", "Test")).unwrap();
        }

        // El mismo Arc en state debe reflejar el cambio
        let g2 = state.graph.lock().unwrap();
        let nodes = g2.search_nodes("Test", 10).unwrap();
        assert!(!nodes.is_empty(), "AppState.graph debe compartir datos con el Arc original");
    }

    /// LlmBackend implementado para Arc<dyn LlmBackend+Send+Sync> delega correctamente.
    #[test]
    fn test_arc_llm_backend_delegates_correctly() {
        let mock: Arc<dyn LlmBackend + Send + Sync + 'static> =
            Arc::new(MockLlmBackend::new(vec!["respuesta delegada"]));

        // Llamar a LlmBackend::complete directamente sobre el Arc<dyn ...>
        let result = mock.complete("system", "user", 128, 0.5).unwrap();
        assert_eq!(result, "respuesta delegada");
    }

    // ── Paso 21: alcance del lock del grafo ──────────────────────────────────

    /// LLM de prueba que, en cada inferencia, comprueba si el lock del grafo
    /// está libre. Con el diseño anterior (lock durante todo el ciclo o toda
    /// la petición) cada llamada lo encontraba tomado.
    #[derive(Clone)]
    struct LockProbeLlm {
        inner: MockLlmBackend,
        graph: Arc<Mutex<MemoryGraph>>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        held:  Arc<std::sync::atomic::AtomicUsize>,
    }

    impl LockProbeLlm {
        fn new(graph: &Arc<Mutex<MemoryGraph>>, responses: Vec<&str>) -> Self {
            Self {
                inner: MockLlmBackend::new(responses),
                graph: Arc::clone(graph),
                calls: Default::default(),
                held:  Default::default(),
            }
        }

        /// (inferencias totales, inferencias con el grafo bloqueado)
        fn report(&self) -> (usize, usize) {
            use std::sync::atomic::Ordering::SeqCst;
            (self.calls.load(SeqCst), self.held.load(SeqCst))
        }
    }

    impl LlmBackend for LockProbeLlm {
        fn complete(&self, system: &str, user: &str, max: i32, temp: f32) -> Result<String, BridgeError> {
            use std::sync::atomic::Ordering::SeqCst;
            self.calls.fetch_add(1, SeqCst);
            if self.graph.try_lock().is_err() {
                self.held.fetch_add(1, SeqCst);
            }
            self.inner.complete(system, user, max, temp)
        }
    }

    #[test]
    fn test_subconscious_cycle_releases_graph_lock_during_inference() {
        let graph = Arc::new(Mutex::new(test_graph()));
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("rust",    "concept", "Rust")).unwrap();
            g.insert_node(&Node::new("ml",      "concept", "ML")).unwrap();
            g.insert_node(&Node::new("memoria", "concept", "Memoria")).unwrap();
            // El muestreo es aleatorio (Paso 23): rust ↔ ml es el único vacío.
            g.insert_edge("rust", "usa", "memoria", 0.9).unwrap();
            g.insert_edge("ml", "usa", "memoria", 0.9).unwrap();
        }
        let llm = LockProbeLlm::new(&graph, vec![
            r#"{"question":"¿Puede Rust ejecutar ML?","reasoning":"Interop técnica."}"#,
            r#"{"status":"VERIFIED","confidence":0.90,"reasoning":"Confirmado."}"#,
            r#"{"triples":[{"subject":"rust","relation":"ejecuta","object":"ml","weight":0.88}]}"#,
        ]);
        let engine = CuriosityEngine::new_silent(
            llm.clone(),
            MockSearchBackend::new(vec![vec!["Rust tiene crates para ML como burn y linfa."]]),
        );

        let report = engine.run_sleep_cycle(&*graph, 1).into_iter().next().unwrap().unwrap();

        assert!(!report.triples_consolidated.is_empty(), "el ciclo debe llegar a la fase de consolidación");
        let (calls, held) = llm.report();
        assert_eq!(calls, 3, "pregunta + validación + síntesis");
        assert_eq!(held, 0, "{held} de {calls} inferencias se ejecutaron con el grafo bloqueado");
    }

    #[test]
    fn test_bridge_releases_graph_lock_during_inference() {
        let graph = Arc::new(Mutex::new(test_graph()));
        graph.lock().unwrap().insert_node(&Node::new("rust", "concept", "Rust")).unwrap();
        let llm = LockProbeLlm::new(&graph, vec![
            r#"{"triples":[{"subject":"rust","relation":"es","object":"lenguaje","weight":0.9}]}"#,
            r#"{"i":"query","q":"¿Qué es Rust?","e":["rust"],"t":"none","p":{}}"#,
            "Rust es un lenguaje de sistemas.",  // síntesis
        ]);
        let bridge = NeuralBridge::new(llm.clone());

        let learned = bridge.extract_and_memorize(&*graph, "Rust es un lenguaje.").unwrap();
        assert_eq!(learned.edges_created, 1, "la escritura final sí debe llegar al grafo");
        let answer = bridge.query_with_memory(&*graph, "¿Qué es Rust?", "").unwrap();
        assert!(answer.contains("Rust"));

        let (calls, held) = llm.report();
        assert_eq!(calls, 3);
        assert_eq!(held, 0, "{held} de {calls} inferencias se ejecutaron con el grafo bloqueado");
    }
}
