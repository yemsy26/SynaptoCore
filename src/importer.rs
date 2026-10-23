// ============================================================
// importer.rs — SynaptoCore Paso 27: Importación por lotes
//
// Carga tripletas de un archivo JSONL directamente en el grafo, sin pasar
// por el LLM (que tarda 1-2 s por texto: semanas para millones de hechos):
//
//   {"subject": "Ecuador", "relation": "tiene_capital", "object": "Quito"}
//   {"subject": "Quito", "relation": "es_capital_de", "object": "Ecuador", "weight": 0.9}
//
// - Cada concepto se resuelve contra el que ya lo nombra (índice
//   `nodes.canon`); los nuevos se vectorizan en bloque, fuera del lock.
// - Cada tramo se escribe en una transacción junto con su progreso: si el
//   proceso se corta, importar el mismo lote continúa donde iba.
// - Las aristas llevan origin = imported y source = lote: no se olvidan y
//   `--deshacer <lote>` las retira sin tocar lo aprendido por otra vía.
//
// Uso (con el servidor parado o en marcha; WAL admite ambos):
//   synaptocore importar paises.jsonl [--lote paises]
//   synaptocore importar --lista
//   synaptocore importar --deshacer paises
// ============================================================

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

use crate::memory_graph::{canonical_id, embed_texts, GraphAccess, MemoryGraph};
use crate::types::Node;

/// Peso de una relación importada que no indica el suyo.
pub const DEFAULT_WEIGHT: f64 = 0.8;

/// Líneas por tramo: una transacción.
const CHUNK_LINES: usize = 1000;

/// Textos por llamada al modelo de embeddings. ONNX reserva memoria para el
/// lote entero: con 1000 nombres por llamada el importador pasaba de 800 MB.
const EMBED_BATCH: usize = 256;

/// Errores de línea que se muestran; el resto solo se cuenta.
const MAX_REPORTED_ERRORS: usize = 5;

#[derive(Deserialize)]
struct ImportLine {
    subject:  String,
    relation: String,
    object:   String,
    #[serde(default)]
    weight:   Option<f64>,
}

/// Tripleta de una línea válida, con los nombres tal como venían.
#[derive(Debug, Clone, PartialEq)]
struct Triple {
    subject:  String,
    relation: String,
    object:   String,
    weight:   f64,
}

/// Progreso y resultado de una importación.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportSummary {
    pub batch:          String,
    /// Línea desde la que se reanudó (0 si el lote empezó ahora).
    pub resumed_from:   u64,
    /// Líneas del archivo procesadas en total (incluidas las del arranque previo).
    pub lines_done:     u64,
    pub invalid_lines:  u64,
    pub nodes_added:    u64,
    pub edges_added:    u64,
    /// Relaciones que el grafo ya tenía.
    pub edges_existing: u64,
    /// Los primeros errores de línea, para mostrarlos.
    pub errors:         Vec<String>,
}

/// Interpreta una línea: `Ok(None)` si está vacía.
fn parse_line(line: &str) -> Result<Option<Triple>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let raw: ImportLine = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let (subject, object) = (raw.subject.trim(), raw.object.trim());
    // La relación sigue el estilo del grafo: snake_case.
    let relation = raw.relation.split_whitespace().collect::<Vec<_>>().join("_");
    if [subject, object, relation.as_str()].iter().any(|s| !s.chars().any(char::is_alphanumeric)) {
        return Err("sujeto, relación y objeto deben tener texto".into());
    }
    if canonical_id(subject) == canonical_id(object) {
        return Err("el sujeto y el objeto son el mismo concepto".into());
    }
    let weight = raw.weight.unwrap_or(DEFAULT_WEIGHT);
    if !(0.0..=1.0).contains(&weight) {
        return Err(format!("peso {weight} fuera de [0, 1]"));
    }
    Ok(Some(Triple { subject: subject.into(), relation, object: object.into(), weight }))
}

/// Importa `path` en el lote `batch`, o lo reanuda si quedó a medias.
/// `progress` se llama tras cada tramo confirmado.
pub fn import_file<G: GraphAccess>(
    graph:    &G,
    path:     &Path,
    batch:    &str,
    progress: &mut dyn FnMut(&ImportSummary),
) -> Result<ImportSummary, String> {
    let file_label = path.display().to_string();
    let db = |e: crate::error::MemoryError| e.to_string();
    let resume_from = match graph.with_graph(|g| g.import_record(batch)).map_err(db)? {
        None => {
            graph.with_graph(|g| g.begin_import(batch, &file_label)).map_err(db)?;
            0
        }
        Some(rec) if rec.finished_at.is_some() => {
            return Err(format!(
                "el lote «{batch}» ya se importó ({} relaciones). Para repetirlo, deshazlo antes con --deshacer {batch}",
                rec.edges_added
            ));
        }
        Some(rec) if rec.file != file_label => {
            return Err(format!("el lote «{batch}» se empezó con otro archivo ({}); usa otro nombre de lote", rec.file));
        }
        Some(rec) => rec.lines_done,
    };

    let reader = BufReader::new(File::open(path).map_err(|e| format!("no se puede abrir {file_label}: {e}"))?);
    let mut lines = reader.lines().enumerate().skip(resume_from as usize);
    let embedder = graph.with_graph(|g| g.embedding_backend());
    let mut summary = ImportSummary {
        batch: batch.into(),
        resumed_from: resume_from,
        lines_done: resume_from,
        ..Default::default()
    };

    loop {
        // ── 1. Leer y validar un tramo ──────────────────────────────
        let mut triples = Vec::new();
        let mut read = 0;
        for (index, line) in lines.by_ref().take(CHUNK_LINES) {
            read += 1;
            let line = line.map_err(|e| format!("error de lectura en la línea {}: {e}", index + 1))?;
            match parse_line(&line) {
                Ok(Some(triple)) => triples.push(triple),
                Ok(None) => {}
                Err(e) => {
                    summary.invalid_lines += 1;
                    if summary.errors.len() < MAX_REPORTED_ERRORS {
                        summary.errors.push(format!("línea {}: {e}", index + 1));
                    }
                }
            }
        }
        if read == 0 {
            break;
        }
        let lines_done = summary.lines_done + read as u64;

        // ── 2. Conceptos: los que ya existen y los nuevos ───────────
        let mut names: Vec<&str> = triples.iter().flat_map(|t| [t.subject.as_str(), t.object.as_str()]).collect();
        names.sort_unstable();
        names.dedup();
        let resolved: Vec<Option<String>> = graph
            .with_graph(|g| names.iter().map(|n| g.resolve_node_id(n)).collect::<Result<_, _>>())
            .map_err(db)?;
        let mut ids: HashMap<&str, String> = HashMap::new();
        let mut new_nodes: HashMap<String, String> = HashMap::new(); // id → nombre
        for (name, existing) in names.iter().zip(resolved) {
            let id = existing.unwrap_or_else(|| {
                let id = canonical_id(name);
                new_nodes.entry(id.clone()).or_insert_with(|| name.to_string());
                id
            });
            ids.insert(name, id);
        }

        // ── 3. Vectores de los nuevos, fuera del lock ───────────────
        let new_names: Vec<String> = new_nodes.values().cloned().collect();
        let vectors: HashMap<String, Vec<f32>> = new_names
            .chunks(EMBED_BATCH)
            .flat_map(|batch| embed_texts(embedder.as_ref(), batch))
            .collect();
        let nodes: Vec<Node> = new_nodes
            .into_iter()
            .map(|(id, name)| {
                let mut node = Node::new(id, "concept", &name);
                node.embedding = vectors.get(&name).cloned();
                node
            })
            .collect();
        let edges: Vec<(String, String, String, f64)> = triples
            .iter()
            .map(|t| (ids[t.subject.as_str()].clone(), t.relation.clone(), ids[t.object.as_str()].clone(), t.weight))
            .collect();

        // ── 4. Escritura del tramo y su progreso, atómica ───────────
        let out = graph
            .with_graph(|g| g.apply_import_chunk(batch, lines_done, &nodes, &edges))
            .map_err(db)?;
        summary.lines_done = lines_done;
        summary.nodes_added += out.nodes_added;
        summary.edges_added += out.edges_added;
        summary.edges_existing += out.edges_existing;
        progress(&summary);
    }

    graph.with_graph(|g| g.finish_import(batch)).map_err(db)?;
    Ok(summary)
}

// ──────────────────────────────────────────────────────────────
//  Línea de comandos
// ──────────────────────────────────────────────────────────────

const USAGE: &str = "Uso:
  synaptocore importar <archivo.jsonl> [--lote <nombre>]
  synaptocore importar --lista
  synaptocore importar --deshacer <lote>

Cada línea del archivo es una tripleta JSON:
  {\"subject\": \"Ecuador\", \"relation\": \"tiene_capital\", \"object\": \"Quito\", \"weight\": 0.9}
(weight es opcional; por defecto 0.8)";

/// Orden de la línea de comandos, ya validada.
#[derive(Debug, PartialEq)]
enum Command {
    List,
    Undo(String),
    Import { path: std::path::PathBuf, batch: String },
}

/// `None` si los argumentos no forman una orden válida.
fn parse_args(args: &[String]) -> Option<Command> {
    match args {
        [flag] if flag == "--lista" => Some(Command::List),
        [flag, batch] if flag == "--deshacer" => Some(Command::Undo(batch.clone())),
        [file] | [file, _, _] if !file.starts_with("--") => {
            let path = Path::new(file);
            let batch = match args {
                [_, flag, name] if flag == "--lote" => name.clone(),
                [_] => path.file_stem()?.to_string_lossy().into_owned(),
                _ => return None,
            };
            (!batch.trim().is_empty()).then(|| Command::Import { path: path.to_path_buf(), batch })
        }
        _ => None,
    }
}

/// `synaptocore importar …`: trabaja sobre `synaptocore_memory.db` del
/// directorio actual, la misma base que usa el servidor. Valida la orden
/// (y que el archivo exista) ANTES de abrir la base: abrirla aplica las
/// migraciones de esquema pendientes.
pub async fn run_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let command = parse_args(args).ok_or(USAGE)?;
    if let Command::Import { path, .. } = &command {
        if !path.is_file() {
            return Err(format!("no existe el archivo {}", path.display()).into());
        }
    }
    let graph = MemoryGraph::open("synaptocore_memory.db")?;
    match command {
        Command::List => {
            let records = graph.import_records()?;
            if records.is_empty() {
                println!("No hay lotes importados.");
            }
            for r in records {
                let state = if r.finished_at.is_some() { "completo" } else { "a medias" };
                println!(
                    "{:<24} {:<9} {:>9} relaciones · {:>8} conceptos nuevos · {} líneas · {}",
                    r.batch, state, r.edges_added, r.nodes_added, r.lines_done, r.file
                );
            }
        }
        Command::Undo(batch) => match graph.undo_import(&batch)? {
            Some((edges, nodes)) => {
                println!("✓ Lote «{batch}» deshecho: {edges} relaciones y {nodes} conceptos sin otras relaciones borrados.");
                println!("  Si SynaptoCore está en marcha, reinícialo para que su índice semántico lo refleje.");
            }
            None => println!("No existe el lote «{batch}» (ver --lista)."),
        },
        Command::Import { path, batch } => {
            let graph = crate::attach_embeddings(graph).await;
            let started = Instant::now();
            let summary = tokio::task::spawn_blocking(move || {
                import_file(&graph, &path, &batch, &mut |s| {
                    let new_lines = (s.lines_done - s.resumed_from) as f64;
                    print!(
                        "\r  {} líneas · {} conceptos nuevos · {} relaciones nuevas ({:.0} líneas/s)   ",
                        s.lines_done, s.nodes_added, s.edges_added,
                        new_lines / started.elapsed().as_secs_f64().max(1e-3)
                    );
                    let _ = std::io::stdout().flush();
                })
            })
            .await??;
            println!();
            if summary.resumed_from > 0 {
                println!("  (reanudado desde la línea {})", summary.resumed_from + 1);
            }
            println!(
                "✓ Lote «{}» importado en {:.1} s: {} conceptos nuevos, {} relaciones nuevas, {} ya conocidas, {} líneas inválidas.",
                summary.batch, started.elapsed().as_secs_f64(), summary.nodes_added, summary.edges_added,
                summary.edges_existing, summary.invalid_lines
            );
            for e in &summary.errors {
                println!("  ⚠ {e}");
            }
            println!("  Si SynaptoCore está en marcha, reinícialo para que su índice semántico incluya lo importado.");
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_graph::{MockEmbeddingBackend, TraversalDirection};
    use crate::types::{EdgeOrigin, Provenance};
    use std::sync::Arc;

    fn write_jsonl(dir: &tempfile::TempDir, name: &str, lines: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    fn graph() -> MemoryGraph {
        MemoryGraph::open(":memory:").unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(8)))
    }

    #[test]
    fn test_parse_args() {
        let args = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        assert_eq!(parse_args(&args("--lista")), Some(Command::List));
        assert_eq!(parse_args(&args("--deshacer paises")), Some(Command::Undo("paises".into())));
        assert_eq!(parse_args(&args("datos/paises.jsonl")),
            Some(Command::Import { path: "datos/paises.jsonl".into(), batch: "paises".into() }));
        assert_eq!(parse_args(&args("x.jsonl --lote erp")),
            Some(Command::Import { path: "x.jsonl".into(), batch: "erp".into() }));
        for bad in ["", "--deshacer", "x.jsonl --otro erp", "--lote erp", "a b c d"] {
            assert_eq!(parse_args(&args(bad)), None, "«{bad}»");
        }
    }

    #[test]
    fn test_parse_line_validates_and_normalizes() {
        let t = parse_line(r#"{"subject": " Ecuador ", "relation": "tiene capital", "object": "Quito"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(t, Triple { subject: "Ecuador".into(), relation: "tiene_capital".into(), object: "Quito".into(), weight: DEFAULT_WEIGHT });
        assert_eq!(parse_line("   ").unwrap(), None);
        assert!(parse_line("no es json").is_err());
        assert!(parse_line(r#"{"subject": "Quito", "relation": "es", "object": "quito"}"#).is_err());
        assert!(parse_line(r#"{"subject": "a", "relation": "r", "object": "b", "weight": 1.5}"#).is_err());
        assert!(parse_line(r#"{"subject": "a", "relation": "--", "object": "b"}"#).is_err());
    }

    #[test]
    fn test_import_resolves_existing_concepts_and_keeps_known_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = graph();
        // Ya sabía, por otra vía, que Quito es la capital de Ecuador.
        g.insert_node(&Node::new("ecuador", "concept", "Ecuador")).unwrap();
        g.insert_node(&Node::new("quito", "concept", "Quito")).unwrap();
        g.insert_edge_from("ecuador", "tiene_capital", "quito", 0.9, &Provenance::new(EdgeOrigin::Taught)).unwrap();

        let path = write_jsonl(&dir, "paises.jsonl", &[
            r#"{"subject": "Ecuador", "relation": "tiene_capital", "object": "Quito"}"#,
            r#"{"subject": "Ecuador", "relation": "limita_con", "object": "Perú", "weight": 0.95}"#,
            "",
            r#"{"subject": "Perú", "relation": "tiene_capital", "object": "Lima"}"#,
            "esto no es json",
        ]);
        let s = import_file(&g, &path, "paises", &mut |_| {}).unwrap();

        assert_eq!((s.lines_done, s.invalid_lines, s.nodes_added, s.edges_added, s.edges_existing), (5, 1, 2, 2, 1));
        assert_eq!(s.errors.len(), 1);
        assert!(s.errors[0].starts_with("línea 5:"), "{}", s.errors[0]);
        assert_eq!(g.get_node("peru").unwrap().unwrap().name, "Perú", "id canónico, nombre tal cual");
        assert_eq!(g.vector_index_len(), 4, "los conceptos nuevos entran vectorizados");

        let hood = g.get_neighbors("ecuador", TraversalDirection::Outgoing).unwrap();
        let origin = |rel: &str| hood.neighbors.iter().find(|(e, _)| e.relation == rel).map(|(e, _)| (e.origin, e.source.clone()));
        assert_eq!(origin("tiene_capital"), Some((EdgeOrigin::Taught, None)), "lo conocido no se toca");
        assert_eq!(origin("limita_con"), Some((EdgeOrigin::Imported, Some("paises".into()))));

        let rec = g.import_record("paises").unwrap().unwrap();
        assert!(rec.finished_at.is_some());
        assert_eq!((rec.nodes_added, rec.edges_added), (2, 2));
        assert!(import_file(&g, &path, "paises", &mut |_| {}).unwrap_err().contains("ya se importó"));
    }

    #[test]
    fn test_import_resumes_after_interruption_and_undo_keeps_other_knowledge() {
        let dir = tempfile::tempdir().unwrap();
        let g = graph();
        let lines: Vec<String> = (0..2500)
            .map(|i| format!(r#"{{"subject": "concepto {i}", "relation": "sigue_a", "object": "concepto {}"}}"#, i + 1))
            .collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let path = write_jsonl(&dir, "cadena.jsonl", &refs);

        // Corte simulado: el proceso muere justo después de confirmar el
        // primer tramo.
        let cut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            import_file(&g, &path, "cadena", &mut |s| {
                if s.lines_done == CHUNK_LINES as u64 {
                    panic!("corte simulado");
                }
            })
        }));
        assert!(cut.is_err());
        let rec = g.import_record("cadena").unwrap().unwrap();
        assert_eq!(rec.lines_done, CHUNK_LINES as u64);
        assert_eq!(rec.finished_at, None);

        let resumed = import_file(&g, &path, "cadena", &mut |_| {}).unwrap();
        assert_eq!(resumed.resumed_from, CHUNK_LINES as u64);
        assert_eq!(g.stats().unwrap().edge_count, 2500, "ni duplicados ni huecos");

        // Algo aprendido por otra vía sobre un concepto del lote sobrevive.
        g.insert_node(&Node::new("cafe", "concept", "café")).unwrap();
        g.insert_edge_from("concepto_7", "inspira", "cafe", 0.9, &Provenance::new(EdgeOrigin::Taught)).unwrap();
        assert_eq!(g.undo_import("cadena").unwrap(), Some((2500, 2500)));
        assert_eq!(g.stats().unwrap().edge_count, 1);
        assert!(g.get_node("concepto_7").unwrap().is_some());
        assert_eq!(g.vector_index_len(), 2, "el índice en RAM sigue a los borrados");
        assert_eq!(g.import_record("cadena").unwrap(), None);
    }
}
