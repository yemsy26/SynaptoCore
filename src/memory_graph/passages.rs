// ============================================================
// memory_graph/passages.rs — SynaptoCore Paso 30: memoria de pasajes
//
// El grafo guarda hechos sueltos (sujeto → relación → objeto); para
// explicar con detalle hace falta el texto del que salieron. Un pasaje es
// un fragmento de texto con su procedencia: una fuente de una
// investigación (título y URL) o lo que el usuario enseña.
//
//   • Se guardan tal cual, troceados a ≤ 500 caracteres (el modelo de
//     embeddings solo lee los primeros ~128 tokens) y sin duplicados.
//   • Al responder desde la memoria, los relevantes para la pregunta van
//     al prompt numerados y la respuesta los cita como fuentes [n]. La
//     búsqueda es híbrida (vector + texto completo con FTS5): el modelo de
//     embeddings no reconoce palabras raras («grafeno» ↔ su pasaje: 0,29).
//   • Se olvidan como las aristas de su misma procedencia, contando desde
//     su último uso: lo enseñado no se olvida.
//   • Nunca se guarda lo que redacta el modelo: solo fuentes y usuario.
// ============================================================

use rusqlite::{params, OptionalExtension};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use super::{
    blob_to_embedding, embed_texts, embedding_to_blob, normalized, singular_words, GraphAccess, MemoryGraph, VectorIndex,
};
use crate::error::MemoryError;
use crate::types::{current_unix_ts, EdgeOrigin};

/// Tamaño máximo de un pasaje. El modelo de embeddings trunca a 128 tokens
/// (~500 caracteres en español): un pasaje más largo se vectorizaría solo
/// por su principio.
pub const MAX_PASSAGE_CHARS: usize = 500;
/// Más corto que esto no explica nada («Ver más», un título suelto).
pub const MIN_PASSAGE_CHARS: usize = 20;
/// Tope de pasajes por texto enseñado (un archivo entero por `sc learn`).
pub const MAX_PASSAGES_PER_TEXT: usize = 40;

// Relevancia (calibrada con `passage_similarity_calibration` y el modelo de
// producción). Coseno pregunta ↔ pasaje: que la responde 0,65-0,82, salvo
// con palabras raras («grafeno», 0,29); solo del tema 0,14-0,49; ajeno
// -0,05-0,19. Un pasaje es relevante si es muy parecido, o si nombra al
// menos la mitad de las palabras de la pregunta sin ser ajeno.
pub const STRONG_SIMILARITY: f32 = 0.55;
const WEAK_SIMILARITY: f32 = 0.20;
const MIN_LEXICAL: f32 = 0.5;
/// Peso de la coincidencia de palabras al ordenar.
const LEXICAL_BONUS: f32 = 0.3;
/// Candidatos que aporta cada búsqueda (vector y texto) antes de puntuar.
const CANDIDATES: usize = 20;

/// Palabras de una pregunta que no dicen de qué trata (búsqueda léxica).
const QUESTION_WORDS: &[&str] = &[
    "que", "como", "cual", "quien", "donde", "cuando", "cuanto", "cuanta", "es", "son", "fue", "era",
    "ser", "esta", "estan", "hay", "se", "su", "lo", "le", "me", "mi", "tu", "te", "nos", "dime",
    "explica", "explicame", "cuentame", "hablame", "sabe", "sabes", "porque", "what", "how", "why",
    "is", "are",
];

/// Un pasaje por guardar.
#[derive(Debug, Clone, PartialEq)]
pub struct PassageDraft {
    pub text:     String,
    /// Título de la fuente; vacío si no tiene (lo enseñado).
    pub title:    String,
    pub url:      Option<String>,
    pub origin:   EdgeOrigin,
    /// Pregunta que llevó a guardarlo, si la hubo.
    pub question: Option<String>,
}

/// Un pasaje guardado.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Passage {
    pub id:         String,
    pub text:       String,
    pub title:      String,
    pub url:        Option<String>,
    pub origin:     EdgeOrigin,
    pub question:   Option<String>,
    pub created_at: i64,
    /// Reloj del olvido: se reinicia cada vez que fundamenta una respuesta.
    pub last_used:  i64,
    pub uses:       u64,
}

impl Passage {
    /// Título con el que se cita: el de la fuente o, si no tiene, su procedencia.
    pub fn display_title(&self) -> String {
        match (self.title.trim(), self.origin) {
            ("", EdgeOrigin::Taught) => "Lo que me enseñaste".into(),
            ("", _) => "Pasaje guardado".into(),
            (title, _) => title.to_string(),
        }
    }

    fn from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id:         r.get(0)?,
            text:       r.get(1)?,
            title:      r.get(2)?,
            url:        r.get(3)?,
            origin:     EdgeOrigin::parse(&r.get::<_, String>(4)?),
            question:   r.get(5)?,
            created_at: r.get(6)?,
            last_used:  r.get(7)?,
            uses:       r.get::<_, i64>(8)? as u64,
        })
    }
}

const PASSAGE_COLUMNS: &str = "id, text, title, url, origin, question, created_at, last_used, uses";

/// Espacios colapsados: la forma con la que se guarda y se deduplica.
pub fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Trocea un texto en pasajes de como mucho `MAX_PASSAGE_CHARS`: párrafos
/// enteros si caben (juntando los cortos seguidos), si no frases y, como
/// último recurso, cortes entre palabras.
pub fn split_into_passages(text: &str) -> Vec<String> {
    let mut units: Vec<String> = Vec::new();
    for paragraph in paragraphs(text) {
        if paragraph.chars().count() <= MAX_PASSAGE_CHARS {
            units.push(paragraph);
            continue;
        }
        for sentence in sentences(&paragraph) {
            if sentence.chars().count() <= MAX_PASSAGE_CHARS {
                units.push(sentence);
            } else {
                units.extend(pack(sentence.split_whitespace().map(String::from)));
            }
        }
    }
    pack(units.into_iter())
        .into_iter()
        .filter(|p| p.chars().count() >= MIN_PASSAGE_CHARS)
        .take(MAX_PASSAGES_PER_TEXT)
        .collect()
}

/// Párrafos separados por líneas en blanco, con los espacios colapsados.
fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in text.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !current.is_empty() {
                out.push(normalize_text(&current.join(" ")));
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    out
}

/// Frases de un párrafo: cortes tras `. ! ? ; …` seguidos de espacio.
fn sentences(paragraph: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | ';' | '…') && chars.peek().map_or(true, |n| n.is_whitespace()) {
            out.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// Junta piezas seguidas mientras quepan en un pasaje.
fn pack(pieces: impl Iterator<Item = String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if !current.is_empty() && current.chars().count() + 1 + piece.chars().count() > MAX_PASSAGE_CHARS {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&piece);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Pasajes de un texto enseñado por el usuario (o de otra procedencia sin URL).
pub fn drafts_from_text(text: &str, origin: EdgeOrigin) -> Vec<PassageDraft> {
    split_into_passages(text)
        .into_iter()
        .map(|text| PassageDraft { text, title: String::new(), url: None, origin, question: None })
        .collect()
}

/// ¿Nombra el texto todas las palabras de la sonda? Así se decide si un
/// pasaje cubre una entidad de la pregunta (la similitud sola no basta,
/// como en `retrieve_context`). Por raíces: en prosa las palabras van
/// flexionadas («descubrimiento» ↔ «descubrió»).
pub fn mentions(text: &str, probe: &str) -> bool {
    let words: HashSet<String> = singular_words(probe).iter().map(|w| stem(w)).collect();
    let have: HashSet<String> = singular_words(text).iter().map(|w| stem(w)).collect();
    !words.is_empty() && words.is_subset(&have)
}

/// Raíz aproximada para la búsqueda léxica: las 5 primeras letras, para
/// que «descubrió» y «descubierto» coincidan.
fn stem(word: &str) -> String {
    word.chars().take(5).collect()
}

/// Raíces de las palabras de la pregunta que dicen de qué trata.
fn topic_stems(query: &str) -> HashSet<String> {
    singular_words(query).iter().filter(|w| !QUESTION_WORDS.contains(&w.as_str())).map(|w| stem(w)).collect()
}

/// Fracción de las raíces de la pregunta que el pasaje nombra.
fn lexical_coverage(stems: &HashSet<String>, passage: &Passage) -> f32 {
    if stems.is_empty() {
        return 0.0;
    }
    let have: HashSet<String> =
        singular_words(&format!("{} {}", passage.title, passage.text)).iter().map(|w| stem(w)).collect();
    stems.iter().filter(|s| have.contains(*s)).count() as f32 / stems.len() as f32
}

/// Consulta FTS5: cualquiera de las raíces, como prefijo («titan»* →
/// titanio). Entre comillas: ningún carácter se lee como operador.
fn fts_query(stems: &HashSet<String>) -> String {
    let mut terms: Vec<String> = stems
        .iter()
        .map(|s| if s.chars().count() >= 3 { format!("\"{s}\"*") } else { format!("\"{s}\"") })
        .collect();
    terms.sort();
    terms.join(" OR ")
}

/// Guarda pasajes vectorizándolos FUERA del lock del grafo (patrón del
/// Paso 21). Devuelve cuántos son nuevos.
pub fn save_passages<G: GraphAccess>(graph: &G, drafts: Vec<PassageDraft>) -> Result<usize, MemoryError> {
    let drafts: Vec<PassageDraft> = drafts
        .into_iter()
        .map(|mut d| {
            d.text = normalize_text(&d.text);
            d
        })
        .filter(|d| d.text.chars().count() >= MIN_PASSAGE_CHARS)
        .collect();
    if drafts.is_empty() {
        return Ok(0);
    }
    let texts: Vec<String> = drafts.iter().map(|d| d.text.clone()).collect();
    let embedder = graph.with_graph(|g| g.embedding_backend());
    let vectors = embed_texts(embedder.as_ref(), &texts);
    graph.with_graph(|g| g.store_passages(&drafts, &vectors))
}

impl MemoryGraph {
    /// Guarda pasajes (ya normalizados) con sus vectores, si los hay. Un
    /// texto repetido no se duplica: solo mejora su procedencia si la nueva
    /// dura más (lo investigado que el usuario enseña pasa a permanente).
    pub fn store_passages(
        &self,
        drafts:  &[PassageDraft],
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<usize, MemoryError> {
        let now = current_unix_ts();
        let mut inserted: Vec<(String, &Vec<f32>)> = Vec::new();
        let mut count = 0;
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut find = tx.prepare_cached("SELECT id, origin FROM passages WHERE text = ?1")?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO passages (id, text, title, url, origin, question, created_at, last_used, uses, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, 0, ?8)",
            )?;
            let mut promote = tx.prepare_cached(
                "UPDATE passages SET origin = ?2, title = ?3, url = ?4, question = COALESCE(?5, question), last_used = ?6
                 WHERE id = ?1",
            )?;
            let mut index_text = tx.prepare_cached("INSERT INTO passages_fts (id, title, text) VALUES (?1, ?2, ?3)")?;
            let mut retitle = tx.prepare_cached("UPDATE passages_fts SET title = ?2 WHERE id = ?1")?;
            for d in drafts {
                let existing: Option<(String, String)> =
                    find.query_row(params![d.text], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
                match existing {
                    Some((id, origin)) => {
                        if d.origin.supersedes(EdgeOrigin::parse(&origin)) {
                            promote.execute(params![id, d.origin.as_str(), d.title.trim(), d.url, d.question, now])?;
                            retitle.execute(params![id, d.title.trim()])?;
                        }
                    }
                    None => {
                        let id = Uuid::new_v4().to_string();
                        let vector = vectors.get(&d.text);
                        insert.execute(params![
                            id, d.text, d.title.trim(), d.url, d.origin.as_str(), d.question, now,
                            vector.map(|v| embedding_to_blob(v)),
                        ])?;
                        index_text.execute(params![id, d.title.trim(), d.text])?;
                        count += 1;
                        if let Some(v) = vector {
                            inserted.push((id, v));
                        }
                    }
                }
            }
        }
        tx.commit()?;
        let mut index = self.passage_index.borrow_mut();
        for (id, v) in inserted {
            index.upsert(&id, v);
        }
        Ok(count)
    }

    /// Los `k` pasajes relevantes para la pregunta, de mayor a menor.
    /// Candidatos: los más parecidos por vector y los que nombran sus
    /// palabras (FTS5). Relevante: muy parecido (≥ `STRONG_SIMILARITY`), o
    /// nombra al menos la mitad de las palabras y no es ajeno. Sin vectores,
    /// solo cuentan las palabras.
    pub fn relevant_passages(
        &self,
        query:     &str,
        query_vec: Option<&[f32]>,
        k:         usize,
    ) -> Result<Vec<(Passage, f32)>, MemoryError> {
        let stems = topic_stems(query);
        let index = self.passage_index.borrow();
        let unit_query = query_vec.filter(|_| index.len() > 0).and_then(normalized);

        let mut candidates: Vec<String> = Vec::new();
        if let Some(q) = &unit_query {
            candidates.extend(index.search(q, CANDIDATES).into_iter().map(|(id, _)| id));
        }
        if !stems.is_empty() {
            let mut stmt = self.conn.prepare_cached(
                "SELECT id FROM passages_fts WHERE passages_fts MATCH ?1 ORDER BY rank LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![fts_query(&stems), CANDIDATES as i64], |r| r.get::<_, String>(0))?;
            for id in rows {
                candidates.push(id?);
            }
        }
        let mut seen = HashSet::new();
        candidates.retain(|id| seen.insert(id.clone()));

        let mut scored = Vec::new();
        for id in candidates {
            let Some(passage) = self.get_passage(&id)? else { continue };
            let lexical = lexical_coverage(&stems, &passage);
            let similarity = unit_query
                .as_ref()
                .and_then(|q| index.vector(&id).map(|v| v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>()));
            let (relevant, score) = match similarity {
                Some(s) => (
                    s >= STRONG_SIMILARITY || (lexical >= MIN_LEXICAL && s >= WEAK_SIMILARITY),
                    s + LEXICAL_BONUS * lexical,
                ),
                None => (lexical >= MIN_LEXICAL, lexical),
            };
            if relevant {
                scored.push((passage, score));
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.id.cmp(&b.0.id)));
        scored.truncate(k);
        Ok(scored)
    }

    pub fn get_passage(&self, id: &str) -> Result<Option<Passage>, MemoryError> {
        Ok(self.conn
            .query_row(&format!("SELECT {PASSAGE_COLUMNS} FROM passages WHERE id = ?1"), params![id], Passage::from_row)
            .optional()?)
    }

    /// Los pasajes más recientes, para revisarlos.
    pub fn recent_passages(&self, limit: usize) -> Result<Vec<Passage>, MemoryError> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {PASSAGE_COLUMNS} FROM passages ORDER BY created_at DESC, rowid DESC LIMIT ?1"
        ))?;
        let rows = stmt.query_map(params![limit as i64], Passage::from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn passage_count(&self) -> Result<u64, MemoryError> {
        let n: i64 = self.conn.query_row("SELECT COUNT(*) FROM passages", [], |r| r.get(0))?;
        Ok(n as u64)
    }

    /// Borra un pasaje (corrección del usuario). `false` si no existía.
    pub fn delete_passage(&self, id: &str) -> Result<bool, MemoryError> {
        self.conn.execute("DELETE FROM passages_fts WHERE id = ?1", params![id])?;
        let deleted = self.conn.execute("DELETE FROM passages WHERE id = ?1", params![id])? > 0;
        self.passage_index.borrow_mut().remove(id);
        Ok(deleted)
    }

    /// Los pasajes que fundamentaron una respuesta: se reinicia su olvido.
    pub fn touch_passages(&self, ids: &[String]) -> Result<usize, MemoryError> {
        let now = current_unix_ts();
        let mut stmt = self.conn.prepare_cached("UPDATE passages SET uses = uses + 1, last_used = ?2 WHERE id = ?1")?;
        let mut count = 0;
        for id in ids {
            count += stmt.execute(params![id, now])?;
        }
        Ok(count)
    }

    /// Olvido (Paso 30): un pasaje vale 1 al usarse y decae con la tasa de
    /// su procedencia; por debajo de `prune_threshold` se borra. Con la vida
    /// media de lo investigado (90 días) y el umbral 0,05, eso son ~13 meses
    /// sin que fundamente ninguna respuesta.
    pub(super) fn prune_stale_passages(
        &self,
        rate_for:        &dyn Fn(EdgeOrigin) -> Option<f64>,
        prune_threshold: f64,
        now:             i64,
    ) -> Result<usize, MemoryError> {
        let rows: Vec<(String, String, i64)> = self.conn
            .prepare("SELECT id, origin, last_used FROM passages")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<Result<_, _>>()?;
        let stale: Vec<String> = rows
            .into_iter()
            .filter(|(_, origin, last_used)| {
                rate_for(EdgeOrigin::parse(origin))
                    .is_some_and(|rate| (-rate * (now - last_used).max(0) as f64).exp() < prune_threshold)
            })
            .map(|(id, ..)| id)
            .collect();
        if stale.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached("DELETE FROM passages WHERE id = ?1")?;
            for id in &stale {
                stmt.execute(params![id])?;
            }
            // Una sola pasada por el índice de texto (`id` no está indexado).
            for chunk in stale.chunks(500) {
                let placeholders = vec!["?"; chunk.len()].join(", ");
                tx.execute(
                    &format!("DELETE FROM passages_fts WHERE id IN ({placeholders})"),
                    rusqlite::params_from_iter(chunk),
                )?;
            }
        }
        tx.commit()?;
        let mut index = self.passage_index.borrow_mut();
        for id in &stale {
            index.remove(id);
        }
        Ok(stale.len())
    }

    /// Pasajes sin vector `(id, texto)`, para `backfill_embeddings`.
    pub fn passages_missing_embedding(&self, limit: usize) -> Result<Vec<(String, String)>, MemoryError> {
        let mut stmt = self.conn.prepare("SELECT id, text FROM passages WHERE embedding IS NULL LIMIT ?1")?;
        let rows = stmt.query_map(params![limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn store_passage_embeddings(&self, items: &[(String, Vec<f32>)]) -> Result<usize, MemoryError> {
        let tx = self.conn.unchecked_transaction()?;
        let mut stored = 0;
        {
            let mut stmt = tx.prepare_cached("UPDATE passages SET embedding = ?1 WHERE id = ?2")?;
            for (id, v) in items {
                stored += stmt.execute(params![embedding_to_blob(v), id])?;
            }
        }
        tx.commit()?;
        let mut index = self.passage_index.borrow_mut();
        for (id, v) in items {
            index.upsert(id, v);
        }
        Ok(stored)
    }

    /// Reconstruye el índice de pasajes desde SQLite.
    pub(super) fn reload_passage_index(&self) -> Result<usize, MemoryError> {
        let mut index = self.passage_index.borrow_mut();
        *index = VectorIndex::new(index.dims);
        let mut stmt = self.conn.prepare("SELECT id, embedding FROM passages WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        for row in rows {
            let (id, blob) = row?;
            index.upsert(&id, &blob_to_embedding(&blob));
        }
        Ok(index.len())
    }
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_graph::{backfill_embeddings, EmbeddingBackend, MockEmbeddingBackend};
    use std::sync::Arc;

    fn draft(text: &str, origin: EdgeOrigin) -> PassageDraft {
        PassageDraft { text: text.into(), title: "Titanio".into(), url: Some("https://es.wikipedia.org/wiki/Titanio".into()), origin, question: None }
    }

    const GREGOR: &str = "El titanio fue descubierto por William Gregor en 1791 en Cornualles.";

    #[test]
    fn test_split_keeps_paragraphs_and_cuts_long_ones() {
        let short = "La sucursal del centro abre a las 9.\n\nCierra a las 18, salvo los domingos.";
        assert_eq!(split_into_passages(short), ["La sucursal del centro abre a las 9. Cierra a las 18, salvo los domingos."],
            "los párrafos cortos seguidos van juntos");

        let sentence = "El titanio es un metal ligero y resistente a la corrosión. ";
        let long = sentence.repeat(20);
        let parts = split_into_passages(&long);
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.chars().count() <= MAX_PASSAGE_CHARS));
        assert!(parts.iter().all(|p| p.ends_with("corrosión.")), "se corta entre frases: {parts:?}");

        let wall = "palabra ".repeat(200);
        assert!(split_into_passages(&wall).iter().all(|p| p.chars().count() <= MAX_PASSAGE_CHARS));
        assert!(split_into_passages("Ver más").is_empty(), "demasiado corto");
        assert!(split_into_passages(&"Una frase suficientemente larga.\n\n".repeat(500)).len() <= MAX_PASSAGES_PER_TEXT);
    }

    #[test]
    fn test_store_dedups_and_promotes_origin() {
        let g = MemoryGraph::open(":memory:").unwrap();
        let none = HashMap::new();
        assert_eq!(g.store_passages(&[draft(GREGOR, EdgeOrigin::Researched)], &none).unwrap(), 1);
        assert_eq!(g.store_passages(&[draft(GREGOR, EdgeOrigin::Researched)], &none).unwrap(), 0, "sin duplicados");
        assert_eq!(g.passage_count().unwrap(), 1);

        g.store_passages(&[draft(GREGOR, EdgeOrigin::Taught)], &none).unwrap();
        let p = &g.recent_passages(10).unwrap()[0];
        assert_eq!(p.origin, EdgeOrigin::Taught, "lo que el usuario confirma pasa a permanente");
        g.store_passages(&[draft(GREGOR, EdgeOrigin::Inferred)], &none).unwrap();
        assert_eq!(g.get_passage(&p.id).unwrap().unwrap().origin, EdgeOrigin::Taught, "nunca empeora");
    }

    #[test]
    fn test_lexical_retrieval_needs_the_topic_words() {
        let g = MemoryGraph::open(":memory:").unwrap();
        save_passages(&g, vec![
            draft(GREGOR, EdgeOrigin::Researched),
            PassageDraft {
                title: "Grafeno".into(),
                ..draft("El grafeno es una lámina de carbono de un átomo de espesor.", EdgeOrigin::Researched)
            },
        ]).unwrap();

        let hits = g.relevant_passages("¿Quién descubrió el titanio?", None, 3).unwrap();
        assert_eq!(hits.len(), 1, "las palabras de la pregunta («quién») no cuentan: {hits:?}");
        assert!(hits[0].0.text.contains("Gregor"));
        assert!(g.relevant_passages("¿Qué es el cobre?", None, 3).unwrap().is_empty());
        let accents = g.relevant_passages("¿quien descubrio el TITANIO?", None, 3).unwrap();
        assert_eq!(accents.len(), 1, "sin tildes ni mayúsculas también");
        assert!(mentions(&hits[0].0.text, "William Gregor"));
        assert!(mentions(&hits[0].0.text, "descubrimiento del titanio"), "por raíces");
        assert!(!mentions(&hits[0].0.text, "grafeno"));
    }

    #[test]
    fn test_semantic_retrieval_and_backfill() {
        let backend: Arc<dyn EmbeddingBackend> = Arc::new(MockEmbeddingBackend::new(8));
        let g = MemoryGraph::open(":memory:").unwrap();
        g.store_passages(&[draft(GREGOR, EdgeOrigin::Researched)], &HashMap::new()).unwrap();
        let g = g.with_embedding(Arc::clone(&backend));
        assert_eq!(backfill_embeddings(&g, 16).unwrap(), 1, "el pasaje sin vector se completa");

        let q = backend.embed(&[GREGOR]).unwrap().remove(0);
        let hits = g.relevant_passages("otra cosa", Some(&q), 3).unwrap();
        assert_eq!(hits.len(), 1, "muy parecido aunque no comparta palabras");
        assert!(hits[0].1 > 0.99);
        assert!(g.delete_passage(&hits[0].0.id).unwrap());
        assert!(g.relevant_passages("otra cosa", Some(&q), 3).unwrap().is_empty(), "el borrado sale del índice");
        assert!(g.relevant_passages("¿Quién descubrió el titanio?", None, 3).unwrap().is_empty(), "y del de texto");
    }

    /// Calibración de `MIN_PASSAGE_SIMILARITY` (llm_bridge) con el modelo
    /// real: pregunta ↔ pasaje que la responde, que solo trata del tema, y
    /// ajeno. Ejecutar con:
    ///   cargo test passage_similarity_calibration -- --ignored --nocapture
    #[cfg(feature = "embeddings")]
    #[test]
    #[ignore]
    fn passage_similarity_calibration() {
        let model = crate::memory_graph::FastEmbedBackend::new().expect("modelo de embeddings");
        let groups = [
            ("responde", [
                ("¿Quién descubrió el titanio?", "El titanio fue descubierto por William Gregor en 1791 en Cornualles, Inglaterra."),
                ("¿Qué es el grafeno?", "El grafeno es una lámina de carbono de un átomo de espesor con una gran resistencia."),
                ("¿A qué hora abre la sucursal del centro?", "La sucursal del centro abre a las 9 y cierra a las 18, salvo los domingos."),
                ("¿Para qué sirve la penicilina?", "La penicilina es un antibiótico que se usa para tratar infecciones causadas por bacterias."),
            ]),
            ("del tema", [
                ("¿Quién descubrió el titanio?", "El titanio es un metal ligero y resistente a la corrosión que se usa en implantes."),
                ("¿Qué es el grafeno?", "El diamante y el grafito son dos formas cristalinas del carbono."),
                ("¿A qué hora abre la sucursal del centro?", "La tienda del centro vende ropa y calzado de temporada."),
                ("¿Para qué sirve la penicilina?", "Alexander Fleming trabajaba en el hospital St Mary de Londres."),
            ]),
            ("ajeno", [
                ("¿Quién descubrió el titanio?", "La sucursal del centro abre a las 9 y cierra a las 18."),
                ("¿Qué es el grafeno?", "La penicilina es un antibiótico que se usa para tratar infecciones."),
                ("¿Cuál es la capital de Japón?", "El grafeno es una lámina de carbono de un átomo de espesor."),
                ("¿Para qué sirve la penicilina?", "Rust es un lenguaje de programación compilado."),
            ]),
        ];
        for (group, pairs) in groups {
            for (q, p) in pairs {
                let v = model.embed(&[q, p]).unwrap();
                println!("{group:>9} {:.3}  {q} ↔ {p}", crate::memory_graph::cosine_similarity(&v[0], &v[1]));
            }
        }
    }

    /// La regla híbrida con vectores conocidos: coseno 1 → relevante aunque
    /// no comparta palabras; 0,3 → solo si nombra el tema; 0,1 → ajeno.
    #[test]
    fn test_hybrid_relevance_rule() {
        let g = MemoryGraph::open(":memory:").unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(4)));
        let vectors = HashMap::from([(GREGOR.to_string(), vec![1.0, 0.0, 0.0, 0.0])]);
        g.store_passages(&[draft(GREGOR, EdgeOrigin::Researched)], &vectors).unwrap();
        let at = |cos: f32| vec![cos, (1.0 - cos * cos).sqrt(), 0.0, 0.0];
        let found = |q: &str, cos: f32| g.relevant_passages(q, Some(&at(cos)), 3).unwrap().len();

        assert_eq!(found("otra cosa", 1.0), 1, "muy parecido");
        assert_eq!(found("¿Quién descubrió el titanio?", 0.3), 1, "nombra el tema y no es ajeno");
        assert_eq!(found("otra cosa", 0.3), 0, "ni parecido ni nombra el tema");
        assert_eq!(found("¿Quién descubrió el titanio?", 0.1), 0, "nombrar no basta si es ajeno");
    }

    #[test]
    fn test_touch_and_forgetting_by_origin() {
        let g = MemoryGraph::open(":memory:").unwrap();
        let none = HashMap::new();
        g.store_passages(&[draft(GREGOR, EdgeOrigin::Researched), draft("Texto que enseñó el usuario sobre la tienda.", EdgeOrigin::Taught)], &none).unwrap();
        let rate = |o: EdgeOrigin| o.half_life_days().map(|d| std::f64::consts::LN_2 / (d * 86_400.0));
        let researched = g.recent_passages(10).unwrap().into_iter().find(|p| p.origin == EdgeOrigin::Researched).unwrap().id;

        g.conn.execute("UPDATE passages SET last_used = last_used - ?1", params![2 * 365 * 86_400]).unwrap();
        g.touch_passages(std::slice::from_ref(&researched)).unwrap();
        assert_eq!(g.prune_stale_passages(&rate, 0.05, current_unix_ts()).unwrap(), 0, "usarlo reinicia su olvido");
        assert_eq!(g.get_passage(&researched).unwrap().unwrap().uses, 1);

        g.conn.execute("UPDATE passages SET last_used = last_used - ?1", params![2 * 365 * 86_400]).unwrap();
        assert_eq!(g.prune_stale_passages(&rate, 0.05, current_unix_ts()).unwrap(), 1, "dos años sin usarlo → se olvida");
        let left: Vec<EdgeOrigin> = g.recent_passages(10).unwrap().into_iter().map(|p| p.origin).collect();
        assert_eq!(left, [EdgeOrigin::Taught], "lo enseñado no se olvida");
    }
}
