// ============================================================
// subconscious.rs — SynaptoCore Paso 4: El Bucle Subconsciente
//
// CONCEPTO ARQUITECTÓNICO:
//   El cerebro humano consolida conocimiento durante el sueño mediante
//   la reactivación de memorias y la búsqueda de patrones entre
//   conceptos aparentemente no relacionados (teoría del "default mode
//   network"). Este módulo implementa el equivalente sintético:
//
//   Un motor de fondo (CuriosityEngine) que:
//     1. Escanea el grafo buscando pares de nodos débilmente conectados
//        ("vacíos de conocimiento").
//     2. Genera una pregunta de investigación para esa brecha.
//     3. Busca evidencia externa (SearchBackend).
//     4. Valida el hallazgo contra el TruthFilter (EpistemicValidator).
//     5. Si supera el filtro, extrae tripletas y consolida en el grafo.
//
// DECISIÓN: ¿Por qué SQLite random sampling en lugar de petgraph BFS?
//   petgraph BFS requiere cargar el grafo completo en RAM. Para un
//   agente cognitivo con >100K nodos esto es prohibitivo en hardware
//   embebido. SQLite permite `ORDER BY RANDOM() LIMIT N` con un cursor
//   de coste O(k) sobre el índice primario, devolviendo k nodos sin
//   materializar todo el grafo. Esta es la ventaja clave de la
//   arquitectura WAL del Paso 1.
//
// DECISIÓN: ¿Por qué `run_sleep_cycle` es sincrónico ahora?
//   El método está diseñado para vivir en `std::thread::spawn` o en
//   un runtime async (tokio). La firma recibe `&self` (no `&mut self`)
//   y todos los writes van a través del `Mutex<Connection>` interno de
//   MemoryGraph, lo que hace que el spawn sea trivial en el futuro:
//
//       let engine = Arc::new(engine);
//       std::thread::spawn(move || engine.run_sleep_cycle(...));
//
//   El demo lo ejecuta sincrónicamente para mantener la salida ordenada.
// ============================================================

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::error::MemoryError;
use crate::llm_bridge::{node_name, BridgeError, ExtractedTriple, LlmBackend};
use crate::memory_graph::{canonical_id, embed_texts, ordered_pair, GraphAccess, MemoryGraph, TraversalDirection};
use crate::prompts;
use crate::truth_filter::{EpistemicValidator, SearchBackend, ValidationResult, ValidationStatus};
use crate::types::{current_unix_ts, Node};

// ── Constantes de operación ──────────────────────────────────────────────────

/// Intervalo de espera entre ciclos de curiosidad del bucle subconsciente.
/// Reducido a 3 segundos para investigación continua y aprendizaje dinámico sin latencia.
pub const DEFAULT_CURIOSITY_INTERVAL_SECS: u64 = 3;

/// Paso 23: techo del backoff del bucle cuando los ciclos no aprenden nada.
pub const MAX_CURIOSITY_INTERVAL_SECS: u64 = 60;

/// Nodos ancla muestreados al azar por ciclo; de cada uno se evalúan sus
/// vecinos semánticos más cercanos como candidatos a vacío.
const GAP_ANCHORS: usize = 8;
const GAP_NEIGHBORS_PER_ANCHOR: usize = 8;

/// Banda de similitud coseno de un vacío (Paso 23). Por debajo, conceptos
/// sin relación (la pregunta sería absurda); por encima, el mismo concepto
/// con otro nombre (una fusión pendiente, no algo que investigar).
/// Calibración del modelo multilingüe del Paso 22: pares relacionados
/// 0,32–0,67 ("implantes médicos ↔ prótesis dentales" 0,54), pares sin
/// relación ≤ 0,23.
pub const GAP_MIN_SIMILARITY: f32 = 0.35;
pub const GAP_MAX_SIMILARITY: f32 = 0.85;

/// Enfriamiento tras un intento improductivo: 30 min, doblando por cada
/// fallo consecutivo hasta 7 días.
const GAP_COOLDOWN_BASE_SECS: i64 = 30 * 60;
const GAP_COOLDOWN_MAX_SECS: i64 = 7 * 24 * 3600;

/// Tras un intento productivo el par también descansa: si no, el motor
/// seguiría extrayendo tripletas del mismo par ciclo tras ciclo.
const GAP_COOLDOWN_PRODUCTIVE_SECS: i64 = 6 * 3600;

/// Temperatura baja para generación de preguntas: queremos especificidad,
/// no creatividad libre.
const CURIOSITY_TEMPERATURE: f32 = 0.2;

/// Tokens máximos para la pregunta de curiosidad (es corta por diseño).
const CURIOSITY_MAX_TOKENS: i32 = 128;

/// Tokens máximos para la síntesis de tripletas finales.
const SYNTHESIS_MAX_TOKENS: i32 = 512;

/// Temperatura casi determinista para síntesis: queremos consistencia.
const SYNTHESIS_TEMPERATURE: f32 = 0.1;

/// Peso propuesto al validador para la hipótesis «A → B».
const DEFAULT_DISCOVERY_WEIGHT: f64 = 0.75;

/// Número máximo de intentos de parseo del JSON de curiosidad.
const MAX_CURIOSITY_PARSE_RETRIES: u8 = 2;

// ── Tipos internos ────────────────────────────────────────────────────────────

/// Respuesta JSON del LLM al prompt de curiosidad.
#[derive(Deserialize, Debug)]
struct CuriosityQuestion {
    question:  String,
    reasoning: String,
}

/// Resultado de un ciclo de sueño completo.
/// Captura métricas de auditoría sin lanzar errores por ciclos infructuosos.
#[derive(Debug, Clone)]
pub struct SleepCycleReport {
    /// Nombre del concepto A del par investigado.
    pub node_a:              String,
    /// Nombre del concepto B del par investigado.
    pub node_b:              String,
    /// Pregunta de investigación generada por el LLM.
    pub question:            String,
    /// Fuentes externas recuperadas por el SearchBackend.
    pub sources_found:       usize,
    /// Resultado de la validación epistémica.
    pub validation_status:   ValidationStatus,
    /// Tripletas que superaron la validación y se insertaron en el grafo.
    pub triples_consolidated: Vec<String>,   // representación "A→rel→B"
    /// Razón de rechazo, si aplica.
    pub rejection_reason:    Option<String>,
}

// ── Error del ciclo ───────────────────────────────────────────────────────────

/// Errores recuperables del CuriosityEngine.
/// Son "blandos": un error en un ciclo no detiene el motor.
#[derive(Debug, thiserror::Error)]
pub enum CuriosityError {
    #[error("El grafo tiene menos de 2 nodos; no hay pares que investigar")]
    GraphTooSmall,

    #[error("Sin vacíos que investigar: los candidatos están conectados, son duplicados o están en enfriamiento")]
    NoEligibleGap,

    #[error("Error al muestrear nodos del grafo: {0}")]
    GraphSample(String),

    #[error("El LLM no pudo formular una pregunta válida tras {0} intentos")]
    QuestionGenerationFailed(u8),

    #[error("El SearchBackend falló: {0}")]
    SearchFailed(String),

    #[error("La síntesis de tripletas no produjo resultados")]
    EmptySynthesis,

    #[error("Error del bridge LLM: {0}")]
    Bridge(#[from] BridgeError),
}

// ── CuriosityEngine ───────────────────────────────────────────────────────────

/// Motor de Curiosidad Artificial: orquesta el ciclo completo de
/// aprendizaje autónomo a partir de vacíos en el grafo de conocimiento.
///
/// Genérico sobre `L: LlmBackend` y `S: SearchBackend` para permitir
/// mocking total en tests sin necesitar red ni GPU.
pub struct CuriosityEngine<L: LlmBackend, S: SearchBackend> {
    llm:       L,
    search:    S,
    /// Umbral de confianza para el validador epistémico.
    /// Se pasa al `EpistemicValidator` interno de cada ciclo.
    /// Default: 0.5 (heredado de TruthFilter).
    /// Paso 23: para consolidar, además, VERIFIED con confianza
    /// estrictamente mayor (ver `consolidation_allowed`).
    pub validation_threshold: f64,
    /// Logger de eventos del ciclo (closure que recibe líneas de texto).
    /// En producción puede redirigir a `log::debug!` o a un canal de canal.
    /// En la demo imprime a stdout para visualizar el "sueño" del agente.
    log_fn: Box<dyn Fn(&str) + Send + Sync>,
}

impl<L: LlmBackend + Clone, S: SearchBackend + Clone> CuriosityEngine<L, S> {
    /// Crea el motor con un logger por defecto (stdout).
    pub fn new(llm: L, search: S) -> Self {
        Self {
            llm,
            search,
            validation_threshold: 0.5,
            log_fn: Box::new(|msg| println!("  [🧠 subconscious] {}", msg)),
        }
    }

    /// Crea el motor con un logger silencioso (útil en tests).
    pub fn new_silent(llm: L, search: S) -> Self {
        Self {
            llm,
            search,
            validation_threshold: 0.5,
            log_fn: Box::new(|_| {}),
        }
    }

    /// Crea con logger personalizado y umbral de validación.
    pub fn with_config(
        llm:       L,
        search:    S,
        threshold: f64,
        log_fn:    impl Fn(&str) + Send + Sync + 'static,
    ) -> Self {
        Self {
            llm,
            search,
            validation_threshold: threshold,
            log_fn: Box::new(log_fn),
        }
    }

    // ── API pública ──────────────────────────────────────────────────────────

    /// Ejecuta N ciclos de sueño consecutivos sobre el grafo.
    ///
    /// Diseñado para ser llamado desde `std::thread::spawn` o un executor
    /// async en producción. Cada ciclo es independiente: un fallo no
    /// interrumpe los siguientes.
    ///
    /// Retorna los reportes de los ciclos que se completaron (con o sin
    /// aprendizaje nuevo).
    ///
    /// `graph` puede ser un `MemoryGraph` (tests) o un `Mutex<MemoryGraph>`
    /// (servidor): en el segundo caso el lock solo se toma en las fases de
    /// lectura y consolidación, nunca durante la inferencia ni la búsqueda.
    pub fn run_sleep_cycle<G: GraphAccess>(
        &self,
        graph: &G,
        cycles: usize,
    ) -> Vec<Result<SleepCycleReport, CuriosityError>> {
        (0..cycles).map(|i| {
            self.log(&format!("── Ciclo {} de {} ──", i + 1, cycles));
            self.run_single_cycle(graph)
        }).collect()
    }

    // ── Implementación interna ───────────────────────────────────────────────

    /// Un ciclo completo: muestra → pregunta → busca → valida → consolida.
    ///
    /// Tres fases con alcance de lock explícito (Paso 21):
    ///   1. Lectura      (lock, ms): vacío de conocimiento + contexto de A y B.
    ///   2. Investigación (sin lock, segundos): LLM, SearXNG, TruthFilter, síntesis.
    ///   3. Consolidación (lock, ms): deduplicación contra el grafo + inserción.
    ///
    /// Paso 23: el resultado se anota en `gap_attempts` para enfriar el par;
    /// sin esa memoria, un par que no produce nada se elegía una y otra vez.
    fn run_single_cycle<G: GraphAccess>(
        &self,
        graph: &G,
    ) -> Result<SleepCycleReport, CuriosityError> {

        // ── FASE 1: Lectura ──────────────────────────────────────────────────
        // PASO A: Encontrar un par de nodos con vacío de conocimiento, y un
        // contexto breve de cada nodo para enriquecer el prompt.
        let (gap, ctx_a, ctx_b, embedder) = graph.with_graph(|g| {
            let gap = self.find_knowledge_gap(g, current_unix_ts())?;
            let ctx_a = self.summarize_neighbors(g, &gap.a);
            let ctx_b = self.summarize_neighbors(g, &gap.b);
            Ok::<_, CuriosityError>((gap, ctx_a, ctx_b, g.embedding_backend()))
        })?;
        match gap.similarity {
            Some(sim) => self.log(&format!("Vacío detectado: «{}» ↔ «{}» (similitud {sim:.2})", gap.a, gap.b)),
            None      => self.log(&format!("Vacío detectado: «{}» ↔ «{}»", gap.a, gap.b)),
        }

        let result = self.investigate(graph, gap.a.clone(), gap.b.clone(), &ctx_a, &ctx_b, embedder);
        self.remember_attempt(graph, &gap.a, &gap.b, &result);
        result
    }

    /// Fases 2 y 3 del ciclo para el par ya elegido.
    fn investigate<G: GraphAccess>(
        &self,
        graph:    &G,
        node_a:   String,
        node_b:   String,
        ctx_a:    &str,
        ctx_b:    &str,
        embedder: Option<std::sync::Arc<dyn crate::memory_graph::EmbeddingBackend>>,
    ) -> Result<SleepCycleReport, CuriosityError> {
        // ── FASE 2: Investigación (sin lock) ────────────────────────────────
        // PASO B: Generar pregunta de investigación.
        let curiosity_q = self.generate_question(&node_a, &node_b, &ctx_a, &ctx_b)?;
        self.log(&format!("Pregunta generada: «{}»", curiosity_q.question));
        self.log(&format!("Razonamiento: {}", curiosity_q.reasoning));

        // PASO C: Buscar evidencia externa.
        let sources = match self.search.search(&curiosity_q.question) {
            Ok(s)  => s,
            Err(e) => {
                // Fail-open: si la búsqueda falla, continuamos con fuentes vacías.
                // El validador devolverá INCONCLUSIVE y probablemente rechazará.
                self.log(&format!("⚠ SearchBackend falló ({e}); procediendo sin fuentes"));
                vec![]
            }
        };
        self.log(&format!("Fuentes recuperadas: {}", sources.len()));

        // PASO D: Validar con el TruthFilter.
        // Construimos el validator por ciclo (es barato, son solo refs a traits).
        // El subject/relation/object que se valida es la tripleta conceptual
        // más abstracta: A → <pregunta comprimida> → B.
        let relation_tag = slugify(&curiosity_q.question);
        let validator = EpistemicValidator::with_threshold(
            self.llm.clone(),
            self.search.clone(),
            self.validation_threshold,
        );
        let validation = validator.validate_triplet_with_sources(
            &node_a,
            &relation_tag,
            &node_b,
            DEFAULT_DISCOVERY_WEIGHT,
            &sources,
        )?;

        self.log(&format!(
            "Validación: {} (confianza={:.2})",
            validation.status.as_str(),
            validation.confidence
        ));

        // PASO E: Si la validación no acepta, reportar hipótesis descartada.
        if !consolidation_allowed(&validation, self.validation_threshold) {
            // Lo que el TruthFilter aceptaría y la política del subconsciente
            // no (INCONCLUSIVE, o VERIFIED en el umbral) se explica en el log.
            let policy = if validation.should_accept() {
                format!(
                    "el subconsciente solo consolida VERIFIED con confianza > {:.2}. ",
                    self.validation_threshold
                )
            } else {
                String::new()
            };
            let reason = format!(
                "Hipótesis descartada — {} (conf={:.2}): {}{}",
                validation.status.as_str(),
                validation.confidence,
                policy,
                validation.reasoning,
            );
            self.log(&format!("❌ {}", reason));
            return Ok(SleepCycleReport {
                node_a,
                node_b,
                question:             curiosity_q.question,
                sources_found:        sources.len(),
                validation_status:    validation.status,
                triples_consolidated: vec![],
                rejection_reason:     Some(reason),
            });
        }

        // PASO E (cont.): Extraer tripletas de las fuentes.
        let raw_triples = self.synthesize_triples(&node_a, &node_b, &sources)?;

        // Paso 22: embeddings de los conceptos nuevos, aún sin lock.
        let mut names: Vec<String> = raw_triples
            .iter()
            .flat_map(|t| [node_name(&t.subject), node_name(&t.object)])
            .collect();
        names.sort();
        names.dedup();
        let vectors = embed_texts(embedder.as_ref(), &names);

        // ── FASE 3: Consolidación ────────────────────────────────────────────
        let consolidated = graph.with_graph(|g| {
            self.consolidate(g, (&node_a, &node_b), &raw_triples, validation.confidence, &vectors)
        });

        if consolidated.is_empty() {
            self.log("⚠ Síntesis completada pero sin tripletas nuevas para el grafo");
        }

        Ok(SleepCycleReport {
            node_a,
            node_b,
            question:             curiosity_q.question,
            sources_found:        sources.len(),
            validation_status:    validation.status,
            triples_consolidated: consolidated,
            rejection_reason:     None,
        })
    }

    // ── Paso A: Detección de vacíos ──────────────────────────────────────────

    /// Elige un par de nodos sin arista directa que merezca investigarse
    /// (Paso 23).
    ///
    /// 1. Muestra `GAP_ANCHORS` nodos al azar. Antes se tomaban siempre los
    ///    más recientes, y el mismo par se repetía indefinidamente.
    /// 2. Con embeddings, los candidatos de cada ancla son sus vecinos
    ///    semánticos dentro de la banda [`GAP_MIN_SIMILARITY`,
    ///    `GAP_MAX_SIMILARITY`] y gana el más similar: conceptos
    ///    relacionados pero aún sin conectar. Sin embeddings, cualquier otra
    ///    ancla vale.
    /// 3. Se descartan pares conectados, en enfriamiento o casi duplicados.
    ///
    /// Sin candidatos: [`CuriosityError::NoEligibleGap`]. Ya no se "refuerza"
    /// un par conectado: era otra fuente de bucles.
    fn find_knowledge_gap(
        &self,
        graph: &MemoryGraph,
        now:   i64,
    ) -> Result<Gap, CuriosityError> {
        let sample_err = |e: MemoryError| CuriosityError::GraphSample(e.to_string());
        let anchors = graph.sample_random_node_ids(GAP_ANCHORS).map_err(sample_err)?;
        if anchors.len() < 2 {
            return Err(CuriosityError::GraphTooSmall);
        }
        let cooling = graph.gap_pairs_cooling_down(now).map_err(sample_err)?;
        let semantic = graph.vector_index_len() >= 2;

        let mut best: Option<Gap> = None;
        for a in &anchors {
            let candidates: Vec<(String, Option<f32>)> = if semantic {
                graph.similar_to_node(a, GAP_NEIGHBORS_PER_ANCHOR)
                    .into_iter()
                    .filter(|(_, sim)| (GAP_MIN_SIMILARITY..=GAP_MAX_SIMILARITY).contains(sim))
                    .map(|(b, sim)| (b, Some(sim)))
                    .collect()
            } else {
                anchors.iter().filter(|b| *b != a).map(|b| (b.clone(), None)).collect()
            };
            if candidates.is_empty() {
                continue;
            }
            let connected: HashSet<String> = graph
                .get_neighbors(a, TraversalDirection::Both)
                .map(|hood| hood.neighbors.into_iter().map(|(_, n)| n.id).collect())
                .unwrap_or_default();

            for (b, similarity) in candidates {
                let better = match &best {
                    None => true,
                    Some(current) => similarity > current.similarity,
                };
                if better
                    && !connected.contains(&b)
                    && !cooling.contains(&ordered_pair(a, &b))
                    && !is_near_duplicate(a, &b)
                {
                    best = Some(Gap { a: a.clone(), b, similarity });
                }
            }
        }
        best.ok_or(CuriosityError::NoEligibleGap)
    }

    /// Anota el resultado del ciclo en `gap_attempts` y enfría el par
    /// (Paso 23). Un fallo del LLM no cuenta: no dice nada del par.
    fn remember_attempt<G: GraphAccess>(
        &self,
        graph:  &G,
        a:      &str,
        b:      &str,
        result: &Result<SleepCycleReport, CuriosityError>,
    ) {
        let outcome = match result {
            Ok(r) if !r.triples_consolidated.is_empty() => "consolidated",
            Ok(r) if r.rejection_reason.is_some()       => "rejected",
            Ok(_)                                        => "no_triples",
            Err(CuriosityError::Bridge(_))               => return,
            Err(_)                                       => "failed",
        };
        let now = current_unix_ts();
        let recorded = graph.with_graph(|g| {
            let attempts = if outcome == "consolidated" { 0 } else { g.gap_attempts(a, b)? + 1 };
            let cooldown = gap_cooldown_secs(attempts);
            g.record_gap_attempt(a, b, attempts, outcome, now, now + cooldown)?;
            Ok::<_, MemoryError>(cooldown)
        });
        match recorded {
            Ok(secs) => self.log(&format!("⏳ Par en enfriamiento durante {} ({outcome})", human_duration(secs))),
            Err(e)   => self.log(&format!("⚠ No se pudo registrar el intento: {e}")),
        }
    }

    // ── Paso B: Generación de pregunta ───────────────────────────────────────

    fn generate_question(
        &self,
        node_a: &str,
        node_b: &str,
        ctx_a:  &str,
        ctx_b:  &str,
    ) -> Result<CuriosityQuestion, CuriosityError> {
        let system  = prompts::curiosity_question_system_prompt();
        let user    = prompts::format_curiosity_user_prompt(node_a, node_b, ctx_a, ctx_b);
        let mut last_err = String::new();

        for attempt in 1..=MAX_CURIOSITY_PARSE_RETRIES {
            // En el segundo intento, añadimos el error anterior al prompt
            // para que el LLM se auto-corrija (mismo patrón que el retry
            // del bridge y el truth_filter).
            let user_with_retry = if attempt > 1 {
                format!("{user}\n\n[Intento anterior falló: {last_err}. Responde SOLO el JSON válido.]")
            } else {
                user.clone()
            };

            let raw = self.llm.complete(
                system,
                &user_with_retry,
                CURIOSITY_MAX_TOKENS,
                CURIOSITY_TEMPERATURE,
            )?;

            match parse_json_loose::<CuriosityQuestion>(&raw) {
                Ok(q) if !q.question.trim().is_empty() => return Ok(q),
                Ok(_)  => last_err = "campo 'question' vacío".to_string(),
                Err(e) => last_err = e.to_string(),
            }
            self.log(&format!("⚠ Reintentando generación de pregunta (intento {})", attempt));
        }

        Err(CuriosityError::QuestionGenerationFailed(MAX_CURIOSITY_PARSE_RETRIES))
    }

    // ── Paso E: Síntesis de tripletas ────────────────────────────────────────

    fn synthesize_triples(
        &self,
        node_a:  &str,
        node_b:  &str,
        sources: &[String],
    ) -> Result<Vec<ExtractedTriple>, CuriosityError> {
        if sources.is_empty() {
            return Err(CuriosityError::EmptySynthesis);
        }

        let system = prompts::curiosity_synthesis_system_prompt(node_a, node_b);
        let sources_text = sources
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");

        let raw = self.llm.complete(
            &system,
            &sources_text,
            SYNTHESIS_MAX_TOKENS,
            SYNTHESIS_TEMPERATURE,
        )?;

        // Parsear el JSON de extracción y filtrar tripletas inválidas.
        #[derive(Deserialize)]
        struct ExtractionBatch { triples: Vec<ExtractedTriple> }

        let batch = parse_json_loose::<ExtractionBatch>(&raw)
            .map_err(|_| CuriosityError::EmptySynthesis)?;

        // Filtrar: solo tripletas donde al menos uno de los nodos sea A o B,
        // y ambos subject/object sean no vacíos. La deduplicación contra el
        // grafo se hace en `consolidate`, ya con el lock tomado.
        //
        // Paso 23: la comparación es por id canónico. El prompt pide
        // snake_case y los ids del chat llevan espacios o mayúsculas
        // ("diagramas de clases"): con igualdad exacta se descartaba todo
        // y el par nunca se cerraba.
        let targets = [canonical_id(node_a), canonical_id(node_b)];
        let valid: Vec<ExtractedTriple> = batch.triples
            .into_iter()
            .filter(|t| {
                let (s, o) = (canonical_id(&t.subject), canonical_id(&t.object));
                !s.is_empty() && !o.is_empty() && s != o
                    && (targets.contains(&s) || targets.contains(&o))
            })
            .collect();

        Ok(valid)
    }

    // ── Fase 3: Consolidación ────────────────────────────────────────────────

    /// Inserta las tripletas sintetizadas con peso ajustado por la confianza
    /// del validador. Las aristas que ya existen se omiten en silencio.
    ///
    /// Se ejecuta bajo el lock del grafo: la comprobación de existencia y la
    /// inserción son atómicas respecto a otros escritores (chat, federación).
    ///
    /// Paso 23: cada extremo se resuelve al nodo que ya nombra ese concepto
    /// (A, B u otro con la misma forma canónica). Así "diagramas_de_clases"
    /// conecta con «diagramas de clases» en vez de crear un duplicado.
    fn consolidate(
        &self,
        graph:      &MemoryGraph,
        (node_a, node_b): (&str, &str),
        triples:    &[ExtractedTriple],
        confidence: f64,
        vectors:    &HashMap<String, Vec<f32>>,
    ) -> Vec<String> {
        let concept = |id: &str| {
            let name = node_name(id);
            let mut node = Node::new(id, "concept", &name);
            node.embedding = vectors.get(&name).cloned();
            node
        };
        let (canon_a, canon_b) = (canonical_id(node_a), canonical_id(node_b));
        let resolve = |text: &str| -> String {
            match canonical_id(text) {
                c if c == canon_a => node_a.to_string(),
                c if c == canon_b => node_b.to_string(),
                _ => graph.resolve_node_id(text).ok().flatten().unwrap_or_else(|| text.trim().to_string()),
            }
        };
        let mut consolidated = Vec::new();

        for triple in triples {
            let (subject, object) = (resolve(&triple.subject), resolve(&triple.object));

            // No reinsertar aristas que ya existen entre estos nodos exactos.
            let exists = graph
                .get_neighbors(&subject, TraversalDirection::Outgoing)
                .map(|h| h.neighbors.iter().any(|(e, n)| {
                    e.relation == triple.relation && n.id == object
                }))
                .unwrap_or(false);
            if exists {
                continue;
            }

            // Insertar nodos si no existen (idempotente: MemoryGraph ignora duplicados).
            let _ = graph.insert_node(&concept(&subject));
            let _ = graph.insert_node(&concept(&object));

            // Peso ajustado por la confianza del validador.
            let final_weight = triple.weight * confidence;
            match graph.insert_edge(&subject, &triple.relation, &object, final_weight) {
                Ok(_) => {
                    let repr = format!("{}→{}→{}", subject, triple.relation, object);
                    self.log(&format!("✓ Consolidada: {} (w={:.3})", repr, final_weight));
                    consolidated.push(repr);
                }
                Err(e) => {
                    // Duplicado o error de integridad — no es fatal.
                    self.log(&format!("⚠ Tripleta ignorada ({})", e));
                }
            }
        }

        consolidated
    }

    // ── Utilidades ────────────────────────────────────────────────────────────

    /// Genera un resumen de texto de los vecinos de un nodo para enriquecer
    /// el prompt de curiosidad. Retorna cadena vacía si el nodo no tiene vecinos.
    fn summarize_neighbors(&self, graph: &MemoryGraph, node_id: &str) -> String {
        graph
            .get_neighbors(node_id, TraversalDirection::Both)
            .ok()
            .map(|hood| {
                hood.neighbors
                    .iter()
                    .take(3)   // máximo 3 vecinos para mantener el prompt corto
                    .map(|(edge, n)| format!("{} {} {}", node_id, edge.relation, n.id))
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default()
    }

    /// Delega al closure de log.
    fn log(&self, msg: &str) {
        (self.log_fn)(msg);
    }
}

// ── Funciones auxiliares de módulo ────────────────────────────────────────────

/// Parsea JSON tolerante a preamble de texto.
///
/// Los modelos pequeños a veces anteponen "```json\n" o texto libre.
/// Esta función busca el primer '{' y el último '}' para extraer
/// solo el objeto JSON.
fn parse_json_loose<T: for<'de> serde::Deserialize<'de>>(
    raw: &str,
) -> Result<T, serde_json::Error> {
    // Intentar parseo directo primero (caso común, más rápido).
    if let Ok(v) = serde_json::from_str(raw.trim()) {
        return Ok(v);
    }
    // Fallback: recortar preamble buscando el objeto JSON embebido.
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if start <= end {
            return serde_json::from_str(&raw[start..=end]);
        }
    }
    // Si no hay '{', intentar parseo directo del trim (producirá error claro).
    serde_json::from_str(raw.trim())
}

/// Convierte una pregunta en formato slug para usarla como `relation` en el grafo.
///
/// "¿Puede Rust ejecutar modelos de ML?" → "puede_rust_ejecutar_modelos_ml"
/// El slug se usa como identificador de la relación en la tripleta de validación.
fn slugify(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .take(5)        // máximo 5 palabras para mantener IDs cortos
        .collect::<Vec<_>>()
        .join("_")
        .to_lowercase()
}

// ── Paso 23: selección de vacíos y ritmo del bucle ───────────────────────────

/// Par elegido por `find_knowledge_gap`. `similarity` es el coseno cuando
/// la selección fue semántica.
#[derive(Debug, Clone, PartialEq)]
struct Gap {
    a:          String,
    b:          String,
    similarity: Option<f32>,
}

/// Mismo concepto con otro nombre, o un compuesto que lo contiene
/// («diagramas de clases» ⊂ «diagramas de clases y diagramas de
/// secuencia»): es una fusión pendiente, no un vacío que investigar.
/// Se compara por palabras para que "ia" no case dentro de "materia".
fn is_near_duplicate(a: &str, b: &str) -> bool {
    let words = |s: &str| -> HashSet<String> {
        canonical_id(s)
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    };
    let (wa, wb) = (words(a), words(b));
    !wa.is_empty() && !wb.is_empty() && (wa.is_subset(&wb) || wb.is_subset(&wa))
}

/// ¿Puede el subconsciente escribir en el grafo lo que validó? (Paso 23)
///
/// Más estricto que `ValidationResult::should_accept`, que también admite
/// INCONCLUSIVE con confianza ≥ umbral (a medio peso). El subconsciente
/// aprende sin supervisión humana y el validador suele devolver
/// INCONCLUSIVE 0,50 cuando duda: en la prueba en vivo, 5 de cada 7 ciclos
/// productivos consolidaban así relaciones dudosas («Exynos 2400 compite_en
/// Galaxy S24»). Ahora hace falta VERIFIED y confianza estrictamente mayor
/// que el umbral.
fn consolidation_allowed(validation: &ValidationResult, threshold: f64) -> bool {
    matches!(validation.status, ValidationStatus::Verified) && validation.confidence > threshold
}

/// Enfriamiento de un par según sus intentos improductivos consecutivos;
/// `0` significa que el último intento sí consolidó conocimiento.
pub fn gap_cooldown_secs(unproductive_attempts: u32) -> i64 {
    if unproductive_attempts == 0 {
        return GAP_COOLDOWN_PRODUCTIVE_SECS;
    }
    let factor = 1i64 << (unproductive_attempts - 1).min(20);
    GAP_COOLDOWN_BASE_SECS.saturating_mul(factor).min(GAP_COOLDOWN_MAX_SECS)
}

/// "30 min", "4 h", "7 d": para los logs del subconsciente.
fn human_duration(secs: i64) -> String {
    match secs {
        s if s < 3600  => format!("{} min", s / 60),
        s if s < 86400 => format!("{} h", s / 3600),
        s              => format!("{} d", s / 86400),
    }
}

/// Intervalo adaptativo entre ciclos del bucle subconsciente (Paso 23).
///
/// Cada ciclo sin conocimiento nuevo duplica la espera hasta `max`; un ciclo
/// productivo, o un cambio en el grafo hecho por otro (el usuario enseña
/// algo), la devuelve a `base`. Con el grafo agotado el motor deja de gastar
/// GPU y consultas a SearXNG cada pocos segundos.
#[derive(Debug, Clone)]
pub struct CycleBackoff {
    base:    Duration,
    max:     Duration,
    current: Duration,
}

impl CycleBackoff {
    pub fn new(base: Duration, max: Duration) -> Self {
        Self { base, max, current: base }
    }

    /// Ritmo por defecto: `DEFAULT_CURIOSITY_INTERVAL_SECS` hasta
    /// `MAX_CURIOSITY_INTERVAL_SECS`.
    pub fn standard() -> Self {
        Self::new(
            Duration::from_secs(DEFAULT_CURIOSITY_INTERVAL_SECS),
            Duration::from_secs(MAX_CURIOSITY_INTERVAL_SECS),
        )
    }

    pub fn base(&self) -> Duration {
        self.base
    }

    /// Espera antes del próximo ciclo.
    pub fn current(&self) -> Duration {
        self.current
    }

    /// Ajusta la espera según el resultado del ciclo.
    pub fn record(&mut self, productive: bool) {
        self.current = if productive { self.base } else { (self.current * 2).min(self.max) };
    }

    /// El grafo cambió por otra vía: vuelve al ritmo base.
    pub fn reset(&mut self) {
        self.current = self.base;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS UNITARIOS
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_bridge::mock::MockLlmBackend;
    use crate::truth_filter::MockSearchBackend;
    use crate::types::Node;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Grafo con un único vacío: rust ↔ machine_learning. El muestreo es
    /// aleatorio (Paso 23), así que los tests fijan el par por construcción.
    fn populated_graph() -> MemoryGraph {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("rust",            "concept", "Rust")).unwrap();
        g.insert_node(&Node::new("machine_learning","concept", "Machine Learning")).unwrap();
        // Arista existente: rust → es_lenguaje_de → programacion
        g.insert_node(&Node::new("programacion",    "concept", "Programación")).unwrap();
        g.insert_edge("rust", "es_lenguaje_de", "programacion", 0.99).unwrap();
        g.insert_edge("machine_learning", "usa", "programacion", 0.9).unwrap();
        g
    }

    /// LLM caído: cada llamada falla.
    #[derive(Clone)]
    struct FailingLlm;

    impl LlmBackend for FailingLlm {
        fn complete(&self, _: &str, _: &str, _: i32, _: f32) -> Result<String, BridgeError> {
            Err(BridgeError::BackendInit("Ollama no responde".into()))
        }
    }

    fn gap_pair(report: &SleepCycleReport) -> (String, String) {
        ordered_pair(&report.node_a, &report.node_b)
    }

    fn make_engine(
        llm_responses: Vec<&str>,
        search_results: Vec<Vec<&str>>,
    ) -> CuriosityEngine<MockLlmBackend, MockSearchBackend> {
        CuriosityEngine::new_silent(
            MockLlmBackend::new(llm_responses),
            MockSearchBackend::new(search_results),
        )
    }

    // ── Tests de utilidades ───────────────────────────────────────────────────

    #[test]
    fn test_slugify_question_to_relation() {
        let s = slugify("¿Puede Rust ejecutar modelos de ML?");
        // Debe ser snake_case, sin puntuación, máx 5 palabras
        assert!(!s.contains('¿'));
        assert!(!s.contains('?'));
        assert!(s.contains('_'));
        // No debe exceder 5 palabras
        assert!(s.split('_').count() <= 5);
    }

    #[test]
    fn test_slugify_preserves_alphanumeric() {
        let s = slugify("rust machine learning");
        assert_eq!(s, "rust_machine_learning");
    }

    #[test]
    fn test_parse_json_loose_direct() {
        let json = r#"{"question": "test?", "reasoning": "reason"}"#;
        let q: CuriosityQuestion = parse_json_loose(json).unwrap();
        assert_eq!(q.question, "test?");
    }

    #[test]
    fn test_parse_json_loose_with_preamble() {
        let json = "Claro, aquí tienes:\n```json\n{\"question\": \"q?\", \"reasoning\": \"r\"}\n```";
        let q: CuriosityQuestion = parse_json_loose(json).unwrap();
        assert_eq!(q.question, "q?");
    }

    // ── Tests del ciclo subconsciente ─────────────────────────────────────────

    /// El ciclo aprende algo nuevo: el grafo crece de X aristas a X+N.
    #[test]
    fn test_sleep_cycle_grows_graph_on_verified() {
        let graph = populated_graph();
        let edges_before = graph.stats().unwrap().edge_count;

        // LLM sequence:
        //   1. Respuesta al prompt de curiosidad (pregunta)
        //   2. Respuesta al validador (VERIFIED)
        //   3. Respuesta a la síntesis de tripletas
        let engine = make_engine(
            vec![
                // Pregunta de curiosidad
                r#"{"question": "¿Puede Rust implementar algoritmos de machine learning?", "reasoning": "Explorar interoperabilidad de sistemas"}"#,
                // Validador epistémico (TruthFilter interno)
                r#"{"status": "VERIFIED", "confidence": 0.88, "reasoning": "Rust tiene crates de ML como burn y linfa."}"#,
                // Síntesis de tripletas
                r#"{"triples": [{"subject": "rust", "relation": "implementa", "object": "machine_learning", "weight": 0.85}]}"#,
            ],
            vec![
                // Resultados de búsqueda para el validador
                vec!["Rust tiene el crate 'burn' para deep learning y 'linfa' para ML clásico."],
                // Resultados para la pregunta de curiosidad (mismo backend consume ambas)
                vec!["Rust y machine learning son compatibles según múltiples proyectos."],
            ],
        );

        let results = engine.run_sleep_cycle(&graph, 1);
        assert_eq!(results.len(), 1);

        let report = results.into_iter().next().unwrap().unwrap();
        assert!(
            matches!(report.validation_status, ValidationStatus::Verified),
            "Se esperaba VERIFIED, obtuvo {:?}", report.validation_status
        );
        assert_eq!(report.rejection_reason, None);

        // El grafo debe haber crecido.
        let edges_after = graph.stats().unwrap().edge_count;
        assert!(
            edges_after > edges_before,
            "El grafo debería haber crecido: {} → {}", edges_before, edges_after
        );
        assert!(!report.triples_consolidated.is_empty());
    }

    /// El ciclo aborta limpiamente cuando el TruthFilter rechaza el hallazgo.
    #[test]
    fn test_sleep_cycle_rejects_contradicted_hypothesis() {
        let graph = populated_graph();
        let edges_before = graph.stats().unwrap().edge_count;

        let engine = make_engine(
            vec![
                // Pregunta de curiosidad
                r#"{"question": "¿Puede el cortisol compilar programas Rust?", "reasoning": "Hipótesis absurda para prueba"}"#,
                // Validador: CONTRADICTED
                r#"{"status": "CONTRADICTED", "confidence": 0.99, "reasoning": "El cortisol es una hormona, no puede compilar código."}"#,
            ],
            vec![
                vec!["El cortisol es una hormona esteroide producida por las glándulas suprarrenales."],
            ],
        );

        let results = engine.run_sleep_cycle(&graph, 1);
        let report = results.into_iter().next().unwrap().unwrap();

        assert!(matches!(report.validation_status, ValidationStatus::Contradicted));
        assert!(report.rejection_reason.is_some());
        assert!(report.triples_consolidated.is_empty(), "No debe consolidar tripletas rechazadas");

        // El grafo NO debe haber crecido.
        let edges_after = graph.stats().unwrap().edge_count;
        assert_eq!(edges_before, edges_after, "Grafo contaminado por hipótesis rechazada");
    }

    /// Con INCONCLUSIVE y confianza baja → hipótesis descartada.
    #[test]
    fn test_sleep_cycle_rejects_inconclusive_low_confidence() {
        let graph = populated_graph();
        let edges_before = graph.stats().unwrap().edge_count;

        let engine = CuriosityEngine::with_config(
            MockLlmBackend::new(vec![
                r#"{"question": "¿Está la memoria relacionada con el cortisol?", "reasoning": "Posible vínculo"}"#,
                r#"{"status": "INCONCLUSIVE", "confidence": 0.20, "reasoning": "Evidencia muy débil."}"#,
            ]),
            MockSearchBackend::empty(),
            0.5, // umbral: 0.5 > 0.20 → rechaza
            |_| {},
        );

        let report = engine.run_sleep_cycle(&graph, 1)
            .into_iter().next().unwrap().unwrap();

        assert!(!report.rejection_reason.is_none() || report.triples_consolidated.is_empty());
        let edges_after = graph.stats().unwrap().edge_count;
        assert_eq!(edges_before, edges_after);
    }

    /// Paso 23: solo VERIFIED con confianza > umbral escribe en el grafo.
    /// INCONCLUSIVE no consolida nunca, aunque el TruthFilter lo aceptaría.
    #[test]
    fn test_only_verified_above_threshold_consolidates() {
        for (status, confidence, consolidates) in [
            ("INCONCLUSIVE", 0.50, false), // el caso observado en vivo
            ("INCONCLUSIVE", 0.90, false),
            ("VERIFIED",     0.50, false), // igual al umbral: no basta
            ("VERIFIED",     0.51, true),
        ] {
            let graph = populated_graph();
            let edges_before = graph.stats().unwrap().edge_count;
            let validation = format!(r#"{{"status": "{status}", "confidence": {confidence}, "reasoning": "Prueba."}}"#);
            let engine = make_engine(
                vec![
                    r#"{"question": "¿Puede Rust hacer ML?", "reasoning": "Interop"}"#,
                    &validation,
                    r#"{"triples": [{"subject": "rust", "relation": "soporta", "object": "machine_learning", "weight": 0.9}]}"#,
                ],
                vec![vec!["Rust tiene crates de ML."]],
            );

            let report = engine.run_sleep_cycle(&graph, 1).pop().unwrap().unwrap();
            let grew = graph.stats().unwrap().edge_count > edges_before;
            assert_eq!(grew, consolidates, "{status} {confidence}");
            assert_eq!(report.rejection_reason.is_none(), consolidates, "{status} {confidence}");
            if !consolidates {
                assert_eq!(graph.gap_attempts("rust", "machine_learning").unwrap(), 1,
                    "{status} {confidence}: el par descartado se enfría");
            }
        }
    }

    /// Múltiples ciclos: el grafo crece acumulativamente y cada ciclo
    /// investiga un par distinto (el primero queda en enfriamiento).
    #[test]
    fn test_multiple_sleep_cycles_accumulate_knowledge() {
        // Dos vacíos: rust ↔ machine_learning y cortisol ↔ memoria.
        let graph = MemoryGraph::open(":memory:").unwrap();
        for id in ["rust", "machine_learning", "cortisol", "memoria"] {
            graph.insert_node(&Node::new(id, "concept", id)).unwrap();
        }
        for (a, b) in [("rust", "cortisol"), ("rust", "memoria"),
                       ("machine_learning", "cortisol"), ("machine_learning", "memoria")] {
            graph.insert_edge(a, "relacionado_con", b, 0.9).unwrap();
        }

        // El orden de los pares es aleatorio: cada síntesis trae las dos
        // tripletas y el filtro se queda con la del par investigado.
        let synthesis = r#"{"triples": [
            {"subject": "rust", "relation": "soporta", "object": "machine_learning", "weight": 0.88},
            {"subject": "cortisol", "relation": "afecta", "object": "memoria", "weight": 0.80}]}"#;
        let engine = make_engine(
            vec![
                r#"{"question": "¿Cómo se relacionan?", "reasoning": "Vínculo posible"}"#,
                r#"{"status": "VERIFIED", "confidence": 0.90, "reasoning": "Confirmado."}"#,
                synthesis,
                r#"{"question": "¿Cómo se relacionan?", "reasoning": "Vínculo posible"}"#,
                r#"{"status": "VERIFIED", "confidence": 0.85, "reasoning": "Confirmado."}"#,
                synthesis,
            ],
            vec![
                vec!["Rust tiene crates de ML."],
                vec!["El cortisol afecta la memoria a largo plazo."],
            ],
        );

        let edges_before = graph.stats().unwrap().edge_count;
        let reports: Vec<_> = engine.run_sleep_cycle(&graph, 2).into_iter().map(Result::unwrap).collect();

        assert_ne!(gap_pair(&reports[0]), gap_pair(&reports[1]), "el segundo ciclo cambia de par");
        assert!(reports.iter().all(|r| r.triples_consolidated.len() == 1));
        assert_eq!(graph.stats().unwrap().edge_count, edges_before + 2);
    }

    // ── Paso 23: selección de vacíos, enfriamiento y filtro canónico ─────────

    /// Un ciclo que no consolida nada enfría su par; sin otro vacío, el
    /// siguiente ciclo descansa en vez de repetirlo (el bucle observado).
    #[test]
    fn test_unproductive_pair_cools_down_instead_of_looping() {
        let graph = populated_graph();
        let engine = make_engine(
            vec![
                r#"{"question": "¿Puede Rust hacer ML?", "reasoning": "Interop"}"#,
                r#"{"status": "VERIFIED", "confidence": 0.95, "reasoning": "Sí."}"#,
                r#"{"triples": []}"#,
            ],
            vec![vec!["Rust tiene crates de ML."]],
        );

        let first = engine.run_sleep_cycle(&graph, 1).pop().unwrap().unwrap();
        assert!(first.triples_consolidated.is_empty());
        assert_eq!(graph.gap_attempts("rust", "machine_learning").unwrap(), 1);

        let second = engine.run_sleep_cycle(&graph, 1).pop().unwrap();
        assert!(matches!(second, Err(CuriosityError::NoEligibleGap)), "{second:?}");
    }

    /// Un LLM caído no dice nada del par: no se enfría.
    #[test]
    fn test_llm_failure_does_not_cool_the_pair() {
        let graph = populated_graph();
        let engine = CuriosityEngine::new_silent(FailingLlm, MockSearchBackend::empty());
        let result = engine.run_sleep_cycle(&graph, 1).pop().unwrap();
        assert!(matches!(result, Err(CuriosityError::Bridge(_))), "{result:?}");
        assert_eq!(graph.gap_attempts("rust", "machine_learning").unwrap(), 0);
        assert!(graph.gap_pairs_cooling_down(current_unix_ts()).unwrap().is_empty());
    }

    /// La síntesis en snake_case se conecta a los nodos existentes aunque sus
    /// ids lleven espacios o mayúsculas, sin crear duplicados.
    #[test]
    fn test_synthesis_resolves_canonical_ids_to_existing_nodes() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("UML", "concept", "UML")).unwrap();
        graph.insert_node(&Node::new("diagramas de clases", "concept", "diagramas de clases")).unwrap();
        let engine = make_engine(
            vec![
                r#"{"question": "¿Qué diagramas incluye UML?", "reasoning": "Modelado"}"#,
                r#"{"status": "VERIFIED", "confidence": 0.90, "reasoning": "Sí."}"#,
                r#"{"triples": [
                    {"subject": "uml", "relation": "incluye", "object": "diagramas_de_clases", "weight": 0.9},
                    {"subject": "Diagramas de Clases", "relation": "se_dibujan_con", "object": "herramientas_case", "weight": 0.8},
                    {"subject": "java", "relation": "usa", "object": "jvm", "weight": 0.9}]}"#,
            ],
            vec![vec!["UML incluye diagramas de clases y de secuencia."]],
        );

        let report = engine.run_sleep_cycle(&graph, 1).pop().unwrap().unwrap();

        assert_eq!(report.triples_consolidated, [
            "UML→incluye→diagramas de clases",
            "diagramas de clases→se_dibujan_con→herramientas_case",
        ], "la tripleta java→jvm no toca el par y se descarta");
        assert_eq!(graph.stats().unwrap().node_count, 3, "solo herramientas_case es un nodo nuevo");
    }

    #[test]
    fn test_gap_selection_uses_semantic_band_and_skips_ineligible_pairs() {
        let graph = MemoryGraph::open(":memory:").unwrap()
            .with_embedding(std::sync::Arc::new(crate::memory_graph::MockEmbeddingBackend::new(3)));
        for (id, v) in [("titanio",          [1.0, 0.0, 0.0]),
                        ("implantes",        [0.6, 0.8, 0.0]),  // coseno 0.60 con titanio
                        ("titanio_puro",     [1.0, 0.0, 0.05]), // 0.999: casi duplicado
                        ("receta_de_cocina", [0.0, 0.0, 1.0])]  // ~0: sin relación
        {
            let mut node = Node::new(id, "concept", id);
            node.embedding = Some(v.to_vec());
            graph.insert_node(&node).unwrap();
        }
        // implantes ↔ titanio_puro (0.60) queda fuera por estar conectado.
        graph.insert_edge("titanio_puro", "se_usa_en", "implantes", 0.9).unwrap();
        let engine = make_engine(vec![], vec![]);
        let now = current_unix_ts();

        let gap = engine.find_knowledge_gap(&graph, now).unwrap();
        assert_eq!(ordered_pair(&gap.a, &gap.b), ordered_pair("titanio", "implantes"));
        assert!((gap.similarity.unwrap() - 0.6).abs() < 1e-3);

        graph.record_gap_attempt("titanio", "implantes", 1, "no_triples", now, now + 60).unwrap();
        assert!(matches!(engine.find_knowledge_gap(&graph, now), Err(CuriosityError::NoEligibleGap)));
        assert!(engine.find_knowledge_gap(&graph, now + 61).is_ok(), "el enfriamiento expira");
    }

    #[test]
    fn test_near_duplicates_are_not_gaps() {
        assert!(is_near_duplicate("diagramas de clases", "diagramas de clases y diagramas de secuencia"));
        assert!(is_near_duplicate("SQLite", "sqlite"));
        assert!(is_near_duplicate("Snapdragon 8 Gen 3", "Qualcomm_Snapdragon_8_Gen_3"));
        assert!(!is_near_duplicate("IA", "materia prima"), "se compara por palabras");
        assert!(!is_near_duplicate("UML", "diagramas de clases"));
    }

    #[test]
    fn test_gap_cooldown_doubles_up_to_a_week() {
        assert_eq!(gap_cooldown_secs(0), GAP_COOLDOWN_PRODUCTIVE_SECS);
        assert_eq!(gap_cooldown_secs(1), 30 * 60);
        assert_eq!(gap_cooldown_secs(2), 60 * 60);
        assert_eq!(gap_cooldown_secs(4), 4 * 60 * 60);
        assert_eq!(gap_cooldown_secs(40), GAP_COOLDOWN_MAX_SECS);
        assert_eq!(human_duration(gap_cooldown_secs(1)), "30 min");
        assert_eq!(human_duration(gap_cooldown_secs(40)), "7 d");
    }

    #[test]
    fn test_cycle_backoff_grows_when_idle_and_resets() {
        let mut b = CycleBackoff::new(Duration::from_secs(3), Duration::from_secs(60));
        for expected in [6, 12, 24, 48, 60, 60] {
            b.record(false);
            assert_eq!(b.current().as_secs(), expected);
        }
        b.record(true);
        assert_eq!(b.current().as_secs(), 3, "un ciclo productivo vuelve al ritmo base");
        b.record(false);
        b.reset();
        assert_eq!(b.current(), b.base());
    }

    /// Si el grafo tiene menos de 2 nodos, el ciclo falla limpiamente.
    #[test]
    fn test_sleep_cycle_fails_gracefully_on_tiny_graph() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("solo_nodo", "concept", "Solo")).unwrap();

        let engine = make_engine(vec![], vec![]);
        let results = engine.run_sleep_cycle(&graph, 1);
        let err = results.into_iter().next().unwrap();
        assert!(err.is_err(), "Debería fallar con grafo de 1 nodo");
    }

    /// `summarize_neighbors` retorna vacío para nodo sin vecinos.
    #[test]
    fn test_summarize_neighbors_empty_for_isolated_node() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("isla", "concept", "Isla")).unwrap();

        let engine = make_engine(vec![], vec![]);
        let summary = engine.summarize_neighbors(&graph, "isla");
        assert!(summary.is_empty());
    }

    /// `summarize_neighbors` retorna texto legible para nodo con vecinos.
    #[test]
    fn test_summarize_neighbors_with_connections() {
        let graph = populated_graph();
        let engine = make_engine(vec![], vec![]);
        let summary = engine.summarize_neighbors(&graph, "rust");
        // rust → es_lenguaje_de → programacion
        assert!(summary.contains("rust"));
        assert!(summary.contains("es_lenguaje_de") || summary.contains("programacion"));
    }
}
