// ============================================================
// research.rs — Aprendizaje de la investigación en caliente (Paso 24)
//
// PROBLEMA:
//   Cuando el chat no sabía algo, respondía "no tengo información" y no
//   aprendía nada: el subconsciente investigaba pares al azar, no lo que
//   el usuario pregunta.
//
// DISEÑO:
//   • El chat investiga en la web y responde citando las fuentes
//     (`NeuralBridge::prepare_research`).
//   • Después, en segundo plano, este módulo convierte ESAS FUENTES en
//     conocimiento: extrae hechos (1 llamada al LLM), valida cada uno
//     contra las mismas fuentes con el TruthFilter y consolida solo los
//     VERIFIED con confianza > umbral, como el subconsciente.
//   • Nunca aprende de la respuesta del propio modelo: solo de las fuentes.
//   • El llamador pasa un LLM que cede el turno al chat (`BackgroundLlm`).
// ============================================================

use serde::Deserialize;

use crate::llm_bridge::{
    node_name, parse_json_response, validate_triple, BridgeError, ExtractedTriple, LlmBackend,
};
use crate::memory_graph::{embed_texts, GraphAccess};
use crate::prompts;
use crate::subconscious::{consolidate_triples, consolidation_allowed};
use crate::truth_filter::{EpistemicValidator, SearchBackend, SearchHit};
use crate::types::{EdgeOrigin, Provenance};

/// Hechos candidatos por investigación: cada uno cuesta una validación.
const MAX_RESEARCH_TRIPLES: usize = 5;

const EXTRACTION_MAX_TOKENS: i32 = 512;
const EXTRACTION_TEMPERATURE: f32 = 0.1;

/// Umbral de confianza para consolidar: el mismo que el subconsciente.
pub const LEARNING_THRESHOLD: f64 = 0.5;

/// Resultado del aprendizaje a partir de una investigación.
#[derive(Debug, Clone, Default)]
pub struct ResearchLearning {
    /// Hechos que el LLM extrajo de las fuentes.
    pub extracted:    usize,
    /// Aristas nuevas en el grafo, como "sujeto→relación→objeto".
    pub consolidated: Vec<String>,
}

/// Extrae, valida y consolida los hechos de las fuentes de una
/// investigación. Las llamadas al LLM corren sin tocar el grafo; solo la
/// consolidación final toma el lock (patrón del Paso 21).
pub fn learn_from_sources<L, S, G>(
    llm:       &L,
    search:    &S,
    graph:     &G,
    question:  &str,
    hits:      &[SearchHit],
    threshold: f64,
    log:       &dyn Fn(&str),
) -> Result<ResearchLearning, BridgeError>
where
    L: LlmBackend + Clone,
    S: SearchBackend + Clone,
    G: GraphAccess,
{
    if hits.is_empty() {
        return Ok(ResearchLearning::default());
    }

    // ── 1. Extracción de hechos de las fuentes ───────────────────
    #[derive(Deserialize)]
    struct Batch {
        #[serde(default)]
        triples: Vec<ExtractedTriple>,
    }
    let raw = llm.complete(
        &prompts::research_extraction_system_prompt(question),
        &prompts::format_sources(hits),
        EXTRACTION_MAX_TOKENS,
        EXTRACTION_TEMPERATURE,
    )?;
    let triples: Vec<ExtractedTriple> = match parse_json_response::<Batch>(&raw) {
        Ok(batch) => batch.triples.into_iter().filter(validate_triple).take(MAX_RESEARCH_TRIPLES).collect(),
        Err(e) => {
            log(&format!("⚠ Extracción de la investigación no interpretable: {e}"));
            return Ok(ResearchLearning::default());
        }
    };

    // ── 2. Validación de cada hecho contra las mismas fuentes ────
    let snippets: Vec<String> = hits.iter().map(|h| h.snippet.clone()).collect();
    let validator = EpistemicValidator::with_threshold(llm.clone(), search.clone(), threshold);
    let mut accepted: Vec<(ExtractedTriple, f64)> = Vec::new();
    for triple in &triples {
        let verdict = validator.validate_triplet_with_sources(
            &triple.subject, &triple.relation, &triple.object, triple.weight, &snippets,
        )?;
        let fact = format!("{} → {} → {}", triple.subject, triple.relation, triple.object);
        if consolidation_allowed(&verdict, threshold) {
            log(&format!("📚 Hecho verificado ({:.2}): {fact}", verdict.confidence));
            accepted.push((triple.clone(), verdict.confidence));
        } else {
            log(&format!("📚 Hecho descartado ({} {:.2}): {fact}", verdict.status.as_str(), verdict.confidence));
        }
    }
    if accepted.is_empty() {
        return Ok(ResearchLearning { extracted: triples.len(), consolidated: Vec::new() });
    }

    // ── 3. Embeddings fuera del lock y consolidación ─────────────
    let mut names: Vec<String> = accepted
        .iter()
        .flat_map(|(t, _)| [node_name(&t.subject), node_name(&t.object)])
        .collect();
    names.sort();
    names.dedup();
    let embedder = graph.with_graph(|g| g.embedding_backend());
    let vectors = embed_texts(embedder.as_ref(), &names);

    // Paso 26: lo verificado con fuentes se recuerda meses y guarda de dónde
    // salió (cada hecho se validó contra todas las fuentes de la búsqueda).
    let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).filter(|u| !u.is_empty()).collect();
    let provenance = Provenance::new(EdgeOrigin::Researched).with_source(urls.join(" "));
    let consolidated = graph.with_graph(|g| {
        accepted
            .iter()
            .flat_map(|(triple, confidence)| {
                consolidate_triples(g, &[], std::slice::from_ref(triple), *confidence, &provenance, &vectors, log)
            })
            .collect()
    });
    Ok(ResearchLearning { extracted: triples.len(), consolidated })
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_bridge::mock::MockLlmBackend;
    use crate::memory_graph::{MemoryGraph, TraversalDirection};
    use crate::truth_filter::MockSearchBackend;
    use crate::types::Node;

    fn hit(snippet: &str) -> SearchHit {
        SearchHit { title: "Titanio".into(), url: "https://es.wikipedia.org/wiki/Titanio".into(), snippet: snippet.into() }
    }

    #[test]
    fn test_learns_only_verified_facts_from_sources() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("titanio", "concept", "titanio")).unwrap();
        let llm = MockLlmBackend::new(vec![
            r#"{"triples": [
                {"subject": "William Gregor", "relation": "descubrio", "object": "titanio", "weight": 0.9},
                {"subject": "titanio", "relation": "es", "object": "metal_pesado", "weight": 0.6}]}"#,
            r#"{"status": "VERIFIED", "confidence": 0.95, "reasoning": "La fuente lo afirma."}"#,
            r#"{"status": "INCONCLUSIVE", "confidence": 0.50, "reasoning": "No consta."}"#,
        ]);

        let learned = learn_from_sources(
            &llm, &MockSearchBackend::empty(), &graph, "¿Quién descubrió el titanio?",
            &[hit("William Gregor descubrió el titanio en 1791.")], LEARNING_THRESHOLD, &|_| {},
        ).unwrap();

        assert_eq!(learned.extracted, 2);
        assert_eq!(learned.consolidated, ["William Gregor→descubrio→titanio"],
            "solo el VERIFIED; el extremo existente se reutiliza por id canónico");
        assert_eq!(graph.stats().unwrap().node_count, 2, "no se crea metal_pesado");

        // Paso 26: consta como investigado, con la fuente de la búsqueda.
        let hood = graph.get_neighbors("titanio", TraversalDirection::Incoming).unwrap();
        let edge = &hood.neighbors[0].0;
        assert_eq!(edge.origin, EdgeOrigin::Researched);
        assert_eq!(edge.source.as_deref(), Some("https://es.wikipedia.org/wiki/Titanio"));
    }

    #[test]
    fn test_without_sources_or_with_bad_json_learns_nothing() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let llm = MockLlmBackend::new(vec!["esto no es json"]);
        let none = learn_from_sources(&llm, &MockSearchBackend::empty(), &graph, "q", &[], 0.5, &|_| {}).unwrap();
        assert!(none.consolidated.is_empty());
        assert_eq!(llm.call_count(), 0, "sin fuentes no se llama al LLM");

        let bad = learn_from_sources(&llm, &MockSearchBackend::empty(), &graph, "q", &[hit("x")], 0.5, &|_| {}).unwrap();
        assert!(bad.consolidated.is_empty());
        assert_eq!(graph.stats().unwrap().edge_count, 0);
    }
}
