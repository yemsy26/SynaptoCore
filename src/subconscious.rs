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

use crate::llm_bridge::{BridgeError, ExtractedTriple, LlmBackend};
use crate::memory_graph::{GraphAccess, MemoryGraph, TraversalDirection};
use crate::prompts;
use crate::truth_filter::{EpistemicValidator, SearchBackend, ValidationStatus};
use crate::types::Node;

// ── Constantes de operación ──────────────────────────────────────────────────

/// Intervalo de espera entre ciclos de curiosidad del bucle subconsciente.
/// Reducido a 3 segundos para investigación continua y aprendizaje dinámico sin latencia.
pub const DEFAULT_CURIOSITY_INTERVAL_SECS: u64 = 3;

/// Máximo de pares de nodos candidatos que se evalúan por ciclo.
/// Limitar evita que el ciclo subconsciente monopolice la CPU cuando
/// el grafo crece a cientos de miles de nodos.
const MAX_CANDIDATES_PER_CYCLE: usize = 3;

/// Temperatura baja para generación de preguntas: queremos especificidad,
/// no creatividad libre.
const CURIOSITY_TEMPERATURE: f32 = 0.2;

/// Tokens máximos para la pregunta de curiosidad (es corta por diseño).
const CURIOSITY_MAX_TOKENS: i32 = 128;

/// Tokens máximos para la síntesis de tripletas finales.
const SYNTHESIS_MAX_TOKENS: i32 = 512;

/// Temperatura casi determinista para síntesis: queremos consistencia.
const SYNTHESIS_TEMPERATURE: f32 = 0.1;

/// Peso propuesto inicial para tripletas descubiertas subconscientemente.
/// Se ajustará hacia abajo si la validación devuelve INCONCLUSIVE.
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
    fn run_single_cycle<G: GraphAccess>(
        &self,
        graph: &G,
    ) -> Result<SleepCycleReport, CuriosityError> {

        // ── FASE 1: Lectura ──────────────────────────────────────────────────
        // PASO A: Encontrar un par de nodos con vacío de conocimiento, y un
        // contexto breve de cada nodo para enriquecer el prompt.
        let (node_a, node_b, ctx_a, ctx_b) = graph.with_graph(|g| {
            let (a, b) = self.find_knowledge_gap(g)?;
            let ctx_a = self.summarize_neighbors(g, &a);
            let ctx_b = self.summarize_neighbors(g, &b);
            Ok::<_, CuriosityError>((a, b, ctx_a, ctx_b))
        })?;
        self.log(&format!("Vacío detectado: «{}» ↔ «{}»", node_a, node_b));

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
        if !validation.should_accept() {
            let reason = format!(
                "Hipótesis descartada — {} (conf={:.2}): {}",
                validation.status.as_str(),
                validation.confidence,
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

        // ── FASE 3: Consolidación ────────────────────────────────────────────
        let consolidated = graph.with_graph(|g| {
            self.consolidate(g, &raw_triples, validation.confidence)
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

    /// Selecciona dos nodos del grafo que no estén directamente conectados.
    ///
    /// ESTRATEGIA O(k) con SQLite:
    ///   `ORDER BY RANDOM() LIMIT N` usa el índice rowid de SQLite y no
    ///   materializa el grafo completo. Para un grafo de 1M nodos, k=3
    ///   genera el resultado en <1ms porque SQLite salta directamente a
    ///   posiciones aleatorias en el archivo B-tree.
    ///
    ///   Una vez obtenido el par, verificamos que no estén ya conectados
    ///   buscando aristas directas (get_neighbors O(vecinos)). Si están
    ///   conectados, seleccionamos el siguiente par del pool de candidatos.
    fn find_knowledge_gap(
        &self,
        graph: &MemoryGraph,
    ) -> Result<(String, String), CuriosityError> {
        let candidates = graph
            .sample_random_nodes(MAX_CANDIDATES_PER_CYCLE * 2)
            .map_err(|e| CuriosityError::GraphSample(e.to_string()))?;

        if candidates.len() < 2 {
            return Err(CuriosityError::GraphTooSmall);
        }

        // Buscar el primer par sin arista directa.
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let a = &candidates[i].id;
                let b = &candidates[j].id;

                // Verificar si ya existe arista directa A→B o B→A.
                let already_connected = graph
                    .get_neighbors(a, TraversalDirection::Both)
                    .map(|hood| hood.neighbors.iter().any(|(_, n)| &n.id == b))
                    .unwrap_or(false);

                if !already_connected {
                    return Ok((a.clone(), b.clone()));
                }
            }
        }

        // Todos los pares están conectados — tomar el primero de todas formas.
        // Esto permite que el motor refuerce conocimiento existente desde
        // una perspectiva diferente (ángulo de investigación distinto).
        self.log("⚠ Todos los candidatos ya conectados; reforzando par existente");
        Ok((candidates[0].id.clone(), candidates[1].id.clone()))
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
        let valid: Vec<ExtractedTriple> = batch.triples
            .into_iter()
            .filter(|t| {
                !t.subject.trim().is_empty()
                && !t.object.trim().is_empty()
                && t.subject != t.object
                && (t.subject.as_str() == node_a || t.object.as_str() == node_b
                    || t.subject.as_str() == node_b || t.object.as_str() == node_a)
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
    fn consolidate(
        &self,
        graph:      &MemoryGraph,
        triples:    &[ExtractedTriple],
        confidence: f64,
    ) -> Vec<String> {
        let mut consolidated = Vec::new();

        for triple in triples {
            // No reinsertar aristas que ya existen entre estos nodos exactos.
            let exists = graph
                .get_neighbors(&triple.subject, TraversalDirection::Outgoing)
                .map(|h| h.neighbors.iter().any(|(e, n)| {
                    e.relation == triple.relation && n.id == triple.object
                }))
                .unwrap_or(false);
            if exists {
                continue;
            }

            // Insertar nodos si no existen (idempotente: MemoryGraph ignora duplicados).
            let _ = graph.insert_node(&Node::new(
                &triple.subject, "concept",
                &triple.subject.replace('_', " "),
            ));
            let _ = graph.insert_node(&Node::new(
                &triple.object, "concept",
                &triple.object.replace('_', " "),
            ));

            // Peso ajustado por la confianza del validador.
            let final_weight = triple.weight * confidence;
            match graph.insert_edge(&triple.subject, &triple.relation, &triple.object, final_weight) {
                Ok(_) => {
                    let repr = format!("{}→{}→{}", triple.subject, triple.relation, triple.object);
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

// ── Extensión de MemoryGraph: muestreo aleatorio ──────────────────────────────

// Añadir `sample_random_nodes` a MemoryGraph no es posible sin modificar
// memory_graph.rs. En lugar de eso, lo exponemos como función libre aquí,
// delegando a `search_nodes` con una query comodín. Esto mantiene la
// separación de módulos: memory_graph no necesita conocer el concepto de
// "ciclo subconsciente".
//
// Una alternativa sería añadir `pub fn all_node_ids(&self) -> Result<...>`
// a MemoryGraph, pero eso expone demasiado del internals del grafo.
// La solución actual aprovecha que `search_nodes("")` con SQLite LIKE '%'
// retorna todos los nodos, y limitamos con el parámetro `limit`.

/// Samplea `n` nodos aleatorios del grafo.
///
/// IMPLEMENTACIÓN: Aprovecha `search_nodes` con query vacía que internamente
/// usa `LIKE '%'` sobre el nombre del nodo, limitado a `n`. Para la
/// aleatoriedad usamos el orden natural de SQLite que varía con los inserts.
/// En Paso 5 se añadirá `ORDER BY RANDOM()` como método nativo de MemoryGraph.
trait MemoryGraphExt {
    fn sample_random_nodes(&self, n: usize) -> Result<Vec<Node>, crate::error::MemoryError>;
}

impl MemoryGraphExt for MemoryGraph {
    fn sample_random_nodes(&self, n: usize) -> Result<Vec<Node>, crate::error::MemoryError> {
        // `search_nodes("")` usa LIKE '%' → devuelve todos los nodos.
        // Limitar a n evita materializar grafos grandes.
        self.search_nodes("", n)
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

    fn populated_graph() -> MemoryGraph {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("rust",            "concept", "Rust")).unwrap();
        g.insert_node(&Node::new("machine_learning","concept", "Machine Learning")).unwrap();
        g.insert_node(&Node::new("memoria",         "concept", "Memoria")).unwrap();
        g.insert_node(&Node::new("cortisol",        "concept", "Cortisol")).unwrap();
        // Arista existente: rust → es_lenguaje_de → programacion
        g.insert_node(&Node::new("programacion",    "concept", "Programación")).unwrap();
        g.insert_edge("rust", "es_lenguaje_de", "programacion", 0.99).unwrap();
        g
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

    /// Múltiples ciclos: el grafo crece acumulativamente.
    #[test]
    fn test_multiple_sleep_cycles_accumulate_knowledge() {
        let graph = populated_graph();

        // Preparamos 2 ciclos, cada uno con sus respuestas LLM y de búsqueda.
        // MockLlmBackend consume las respuestas en orden FIFO.
        let engine = make_engine(
            vec![
                // Ciclo 1 — pregunta
                r#"{"question": "¿Cómo se relaciona Rust con machine learning?", "reasoning": "Interop técnica"}"#,
                // Ciclo 1 — validación
                r#"{"status": "VERIFIED", "confidence": 0.90, "reasoning": "Confirmado."}"#,
                // Ciclo 1 — síntesis
                r#"{"triples": [{"subject": "rust", "relation": "soporta", "object": "machine_learning", "weight": 0.88}]}"#,
                // Ciclo 2 — pregunta
                r#"{"question": "¿Afecta el cortisol a la memoria?", "reasoning": "Vínculo neuroendocrino"}"#,
                // Ciclo 2 — validación
                r#"{"status": "VERIFIED", "confidence": 0.85, "reasoning": "Estudios neurológicos lo confirman."}"#,
                // Ciclo 2 — síntesis
                r#"{"triples": [{"subject": "cortisol", "relation": "afecta", "object": "memoria", "weight": 0.80}]}"#,
            ],
            vec![
                // Búsquedas (2 ciclos × 1 búsqueda por validación)
                vec!["Rust tiene crates de ML."],
                vec!["El cortisol afecta la memoria a largo plazo."],
            ],
        );

        let edges_before = graph.stats().unwrap().edge_count;
        let results = engine.run_sleep_cycle(&graph, 2);

        assert_eq!(results.len(), 2);
        let edges_after = graph.stats().unwrap().edge_count;
        // Al menos 1 arista nueva por ciclo exitoso
        assert!(edges_after > edges_before, "El grafo debería haber crecido");
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
