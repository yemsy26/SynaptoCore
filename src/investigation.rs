// ============================================================
// investigation.rs — SynaptoCore Paso 28: Investigación profunda
//
// El usuario da un objetivo («¿qué ERP open source sirven para farmacias
// en Ecuador?») y el sistema trabaja en segundo plano, por rondas:
//
//   1. Preguntas  — divide el objetivo; en rondas siguientes pregunta por
//                   lo que sigue sin resolver.
//   2. Búsqueda   — SearXNG; cada fuente recibe un número estable [n].
//   3. Hipótesis  — respuestas posibles que las fuentes puedan confirmar
//                   o refutar.
//   4. Evidencia  — búsqueda dirigida por hipótesis y veredicto (apoya /
//                   contradice / insuficiente) citando fuentes. Sin una
//                   fuente citada, ningún veredicto cuenta.
//   5. Para cuando una hipótesis queda apoyada con certeza o se agotan las
//      rondas; redacta la conclusión y aprende lo verificado.
//
// Al modelo de 2B se le pide una cosa pequeña cada vez, con JSON de
// claves cortas. El informe (hipótesis, veredictos, fuentes) lo arma este
// módulo; el modelo solo redacta la conclusión. El estado completo pasa
// por `checkpoint` tras cada paso: el servidor lo guarda (la interfaz lo
// sigue en vivo y sobrevive a un reinicio) y por ahí llega la cancelación.
// ============================================================

use std::collections::HashSet;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::llm_bridge::{parse_json_response, BridgeError, LlmBackend};
use crate::memory_graph::{canonical_id, GraphAccess};
use crate::prompts;
use crate::research;
use crate::truth_filter::{SearchBackend, SearchHit};
use crate::types::current_unix_ts;

/// Rondas por defecto y máximas.
pub const DEFAULT_ROUNDS: u32 = 3;
pub const MAX_ROUNDS: u32 = 5;

/// Esperas antes de reintentar una ronda cuyas búsquedas fallaron todas
/// (motores de SearXNG limitados, red caída).
pub const SEARCH_RETRY_WAITS: [Duration; 2] = [Duration::from_secs(20), Duration::from_secs(60)];

/// Certeza mínima para que «apoya» o «contradice» cuenten. En la prueba en
/// vivo el modelo dio «apoya» con certeza 0,0 o 0,5 a hipótesis que las
/// fuentes no sostenían.
const MIN_VERDICT_CONFIDENCE: f64 = 0.6;

/// Certeza con la que una hipótesis apoyada resuelve la investigación.
pub const ACCEPT_CONFIDENCE: f64 = 0.7;

const MAX_SOURCES: usize = 40;
const HITS_PER_QUERY: usize = 4;
const MAX_HYPOTHESES: usize = 5;
/// Fuentes que ve el juez por hipótesis (contexto corto para 2B).
const MAX_EVIDENCE: usize = 8;
/// Fuentes de las que se aprende al final.
const MAX_LEARNING_SOURCES: usize = 5;
const SNIPPET_CHARS: usize = 300;

const PLAN_TOKENS: i32 = 256;
const JUDGE_TOKENS: i32 = 256;
const CONCLUSION_TOKENS: i32 = 450;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvestigationStatus {
    Running,
    Done,
    Cancelled,
    Failed,
    /// El servidor se reinició con la investigación a medias.
    Interrupted,
}

impl InvestigationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running     => "running",
            Self::Done        => "done",
            Self::Cancelled   => "cancelled",
            Self::Failed      => "failed",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pending,
    Supported,
    Refuted,
    /// Las fuentes no bastan (o el juez no citó ninguna).
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// Número estable con el que se cita: [n].
    pub n:       usize,
    pub title:   String,
    pub url:     String,
    pub snippet: String,
    pub round:   u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub text:          String,
    pub round:         u32,
    pub verdict:       Verdict,
    pub confidence:    f64,
    /// Fuentes que la apoyan / contradicen, por número.
    pub supporting:    Vec<usize>,
    pub contradicting: Vec<usize>,
    pub reasoning:     String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Investigation {
    pub id:          String,
    pub objective:   String,
    pub status:      InvestigationStatus,
    pub max_rounds:  u32,
    pub round:       u32,
    /// Qué está haciendo ahora, para la interfaz.
    pub stage:       String,
    pub questions:   Vec<String>,
    pub sources:     Vec<Source>,
    pub hypotheses:  Vec<Hypothesis>,
    pub conclusion:  Option<String>,
    /// Hechos que pasaron al grafo («sujeto→relación→objeto»).
    pub learned:     Vec<String>,
    pub error:       Option<String>,
    pub created_at:  i64,
    pub updated_at:  i64,
    pub finished_at: Option<i64>,
}

impl Investigation {
    pub fn new(objective: &str, max_rounds: u32) -> Self {
        let now = current_unix_ts();
        Self {
            id:          uuid::Uuid::new_v4().to_string(),
            objective:   objective.trim().to_string(),
            status:      InvestigationStatus::Running,
            max_rounds:  max_rounds.clamp(1, MAX_ROUNDS),
            round:       0,
            stage:       "En cola".into(),
            questions:   Vec::new(),
            sources:     Vec::new(),
            hypotheses:  Vec::new(),
            conclusion:  None,
            learned:     Vec::new(),
            error:       None,
            created_at:  now,
            updated_at:  now,
            finished_at: None,
        }
    }

    /// Cierra la investigación con un estado final.
    pub fn finish(&mut self, status: InvestigationStatus, stage: &str) {
        let now = current_unix_ts();
        self.status = status;
        self.stage = stage.into();
        self.updated_at = now;
        self.finished_at = Some(now);
    }

    /// Registra fuentes nuevas (sin repetir URL) y devuelve los números de
    /// todas las recibidas, también las que ya estaban.
    fn add_sources(&mut self, hits: &[SearchHit], round: u32) -> Vec<usize> {
        let mut numbers = Vec::new();
        for hit in hits {
            let key = if hit.url.is_empty() { &hit.snippet } else { &hit.url };
            let existing = self.sources.iter().find(|s| if s.url.is_empty() { &s.snippet == key } else { &s.url == key });
            match existing {
                Some(s) => numbers.push(s.n),
                None if self.sources.len() < MAX_SOURCES => {
                    let n = self.sources.len() + 1;
                    self.sources.push(Source {
                        n,
                        title:   hit.title.trim().to_string(),
                        url:     hit.url.clone(),
                        snippet: hit.snippet.chars().take(SNIPPET_CHARS).collect::<String>().trim().to_string(),
                        round,
                    });
                    numbers.push(n);
                }
                None => {}
            }
        }
        numbers
    }

    /// Fuentes numeradas para un prompt: "[n] título — fragmento".
    fn format_sources(&self, numbers: &[usize]) -> String {
        numbers
            .iter()
            .filter_map(|n| self.sources.get(n - 1))
            .map(|s| match s.title.as_str() {
                "" => format!("[{}] {}", s.n, s.snippet),
                title => format!("[{}] {} — {}", s.n, title, s.snippet),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn hypotheses_summary(&self) -> String {
        self.hypotheses
            .iter()
            .map(|h| format!("- «{}»: {} ({:.2})", h.text, verdict_label(h.verdict), h.confidence))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// ¿Alguna hipótesis apoyada con certeza suficiente?
    fn resolved(&self) -> bool {
        self.hypotheses
            .iter()
            .any(|h| h.verdict == Verdict::Supported && h.confidence >= ACCEPT_CONFIDENCE)
    }
}

fn verdict_label(v: Verdict) -> &'static str {
    match v {
        Verdict::Pending   => "pendiente",
        Verdict::Supported => "apoyada",
        Verdict::Refuted   => "refutada",
        Verdict::Uncertain => "sin confirmar",
    }
}

/// Tras cada paso el motor entrega el estado y un mensaje; si la función
/// devuelve `false` (cancelada), el motor se detiene.
pub type Checkpoint<'a> = dyn FnMut(&Investigation, &str) -> bool + 'a;

/// Avanza de etapa y consulta si se puede seguir.
fn advance(inv: &mut Investigation, checkpoint: &mut Checkpoint, stage: &str, message: &str) -> bool {
    inv.stage = stage.into();
    inv.updated_at = current_unix_ts();
    if checkpoint(inv, message) {
        return true;
    }
    inv.finish(InvestigationStatus::Cancelled, "Cancelada");
    false
}

#[derive(Deserialize, Default)]
struct Questions {
    #[serde(default)]
    q: Vec<String>,
}

#[derive(Deserialize, Default)]
struct Proposals {
    #[serde(default)]
    h: Vec<String>,
}

#[derive(Deserialize)]
struct Judgment {
    v: String,
    #[serde(default)]
    c: f64,
    #[serde(default)]
    a: Vec<i64>,
    #[serde(default)]
    x: Vec<i64>,
    #[serde(default)]
    r: String,
}

/// JSON del modelo, o `None` si no es interpretable.
fn ask_json<T: DeserializeOwned, L: LlmBackend>(
    llm:    &L,
    system: &str,
    user:   &str,
    schema: &serde_json::Value,
    tokens: i32,
    temp:   f32,
) -> Result<Option<T>, BridgeError> {
    let raw = llm.complete_json(system, user, schema, tokens, temp)?;
    Ok(parse_json_response::<T>(&raw).ok())
}

/// Textos nuevos, sin vacíos ni repetidos (ignorando mayúsculas), hasta `limit`.
fn fresh(candidates: Vec<String>, known: &[String], limit: usize) -> Vec<String> {
    let mut seen: HashSet<String> = known.iter().map(|k| k.trim().to_lowercase()).collect();
    candidates
        .into_iter()
        .map(|c| c.trim().to_string())
        .filter(|c| c.chars().any(char::is_alphanumeric) && seen.insert(c.to_lowercase()))
        .take(limit)
        .collect()
}

/// Busca cada pregunta y registra sus fuentes: (números de la ronda, errores).
fn search_round<S: SearchBackend>(search: &S, inv: &mut Investigation, questions: &[String], round: u32) -> (Vec<usize>, Vec<String>) {
    let mut numbers = Vec::new();
    let mut failures = Vec::new();
    for q in questions {
        match search_hits(search, q) {
            Ok(hits) => numbers.extend(inv.add_sources(&hits, round)),
            Err(e) => failures.push(e),
        }
    }
    numbers.sort_unstable();
    numbers.dedup();
    (numbers, failures)
}

fn search_hits<S: SearchBackend>(search: &S, query: &str) -> Result<Vec<SearchHit>, String> {
    search
        .search_detailed(query)
        .map(|hits| hits.into_iter().filter(|h| !h.snippet.trim().is_empty()).take(HITS_PER_QUERY).collect())
        .map_err(|e| e.to_string())
}

/// Investiga `inv.objective` hasta resolverlo o agotar las rondas.
///
/// Devuelve `Err` solo si el LLM falla. Si todas las búsquedas de una ronda
/// fallan, espera cada uno de `retry_waits` y reintenta; después sigue con
/// lo que tenga (o termina «sin fuentes» si no tiene ninguna).
pub fn run<L, S, G>(
    llm:         &L,
    search:      &S,
    graph:       &G,
    inv:         &mut Investigation,
    retry_waits: &[Duration],
    checkpoint:  &mut Checkpoint,
) -> Result<(), BridgeError>
where
    L: LlmBackend + Clone,
    S: SearchBackend + Clone,
    G: GraphAccess,
{
    for round in 1..=inv.max_rounds {
        inv.round = round;

        // ── 1. Preguntas ─────────────────────────────────────────────
        if !advance(inv, checkpoint, "Planificando preguntas", &format!("Ronda {round}: planificando preguntas")) {
            return Ok(());
        }
        let (system, user, limit) = if round == 1 {
            (prompts::investigation_questions_system_prompt(), format!("Objetivo: {}", inv.objective), 4)
        } else {
            let user = format!(
                "Objetivo: {}\n\nPreguntas ya hechas:\n- {}\n\nHipótesis:\n{}",
                inv.objective,
                inv.questions.join("\n- "),
                inv.hypotheses_summary()
            );
            (prompts::investigation_refine_system_prompt(), user, 3)
        };
        let proposed: Questions = ask_json(llm, system, &user, &prompts::investigation_questions_schema(), PLAN_TOKENS, 0.3)?
            .unwrap_or_default();
        let mut questions = fresh(proposed.q, &inv.questions, limit);
        if questions.is_empty() {
            if round > 1 {
                advance(inv, checkpoint, "Sin preguntas nuevas", "No quedan preguntas nuevas que hacer");
                break;
            }
            questions.push(inv.objective.clone());
        }
        inv.questions.extend(questions.iter().cloned());

        // ── 2. Búsqueda (con reintentos si el buscador no responde) ──
        let mut waits = retry_waits.iter();
        let (round_sources, failures) = loop {
            let (numbers, failures) = search_round(search, inv, &questions, round);
            match waits.next() {
                Some(wait) if numbers.is_empty() && !failures.is_empty() => {
                    let message = format!("⚠ El buscador no responde ({}); reintento en {} s", failures[0], wait.as_secs());
                    if !advance(inv, checkpoint, "Esperando al buscador", &message) {
                        return Ok(());
                    }
                    std::thread::sleep(*wait);
                }
                _ => break (numbers, failures),
            }
        };
        let mut message = format!("Ronda {round}: {} pregunta(s), {} fuente(s)", questions.len(), round_sources.len());
        if !failures.is_empty() {
            message.push_str(&format!(" · ⚠ {} búsqueda(s) fallida(s): {}", failures.len(), failures[0]));
        }
        if !advance(inv, checkpoint, "Buscando fuentes", &message) {
            return Ok(());
        }
        if inv.sources.is_empty() {
            inv.error = Some(match failures.first() {
                Some(e) => format!("La búsqueda web falló: {e}"),
                None => "La búsqueda web no devolvió fuentes".into(),
            });
            inv.finish(InvestigationStatus::Failed, "Sin fuentes");
            checkpoint(inv, "⚠ Sin fuentes: no se puede investigar");
            return Ok(());
        }

        // ── 3. Hipótesis (si no quedan abiertas) ─────────────────────
        let open = inv.hypotheses.iter().any(|h| matches!(h.verdict, Verdict::Pending | Verdict::Uncertain));
        if !open && inv.hypotheses.len() < MAX_HYPOTHESES {
            let evidence: Vec<usize> = round_sources.iter().copied().take(MAX_EVIDENCE).collect();
            let known: Vec<String> = inv.hypotheses.iter().map(|h| h.text.clone()).collect();
            let user = format!(
                "Objetivo: {}\n\nHipótesis ya evaluadas:\n{}\n\nFuentes:\n{}",
                inv.objective,
                if known.is_empty() { "(ninguna)".to_string() } else { inv.hypotheses_summary() },
                inv.format_sources(&evidence)
            );
            let proposals: Proposals = ask_json(
                llm, prompts::investigation_hypotheses_system_prompt(), &user,
                &prompts::investigation_hypotheses_schema(), PLAN_TOKENS, 0.5,
            )?
            .unwrap_or_default();
            let statements = proposals.h.into_iter().filter(|h| !h.trim_start().starts_with('¿') && !h.trim_end().ends_with('?')).collect();
            for text in fresh(statements, &known, MAX_HYPOTHESES - inv.hypotheses.len()) {
                inv.hypotheses.push(Hypothesis {
                    text,
                    round,
                    verdict:       Verdict::Pending,
                    confidence:    0.0,
                    supporting:    Vec::new(),
                    contradicting: Vec::new(),
                    reasoning:     String::new(),
                });
            }
            let added = inv.hypotheses.iter().filter(|h| h.round == round).count();
            if !advance(inv, checkpoint, "Formulando hipótesis", &format!("Ronda {round}: {added} hipótesis nueva(s)")) {
                return Ok(());
            }
        }

        // ── 4. Evidencia y veredicto de cada hipótesis abierta ───────
        for i in 0..inv.hypotheses.len() {
            if !matches!(inv.hypotheses[i].verdict, Verdict::Pending | Verdict::Uncertain) {
                continue;
            }
            let text = inv.hypotheses[i].text.clone();
            // La primera vez, búsqueda dirigida; después, lo que trajo la ronda.
            let first_time = inv.hypotheses[i].verdict == Verdict::Pending;
            let mut evidence = Vec::new();
            if first_time {
                if let Ok(hits) = search_hits(search, &text) {
                    evidence = inv.add_sources(&hits, round);
                }
            }
            for n in &round_sources {
                if !evidence.contains(n) {
                    evidence.push(*n);
                }
            }
            evidence.truncate(MAX_EVIDENCE);
            // Sin fuentes nuevas, el veredicto anterior se mantiene.
            if evidence.is_empty() && !first_time {
                continue;
            }
            if !advance(inv, checkpoint, "Buscando evidencias", &format!("Evaluando «{text}»")) {
                return Ok(());
            }

            let judged = if evidence.is_empty() {
                None
            } else {
                let user = format!(
                    "Objetivo: {}\nHipótesis: «{text}»\n\nFuentes:\n{}",
                    inv.objective,
                    inv.format_sources(&evidence)
                );
                ask_json::<Judgment, _>(
                    llm, prompts::investigation_judge_system_prompt(), &user,
                    &prompts::investigation_judge_schema(), JUDGE_TOKENS, 0.1,
                )?
            };
            apply_judgment(&mut inv.hypotheses[i], judged, &evidence);
            if let Some(names) = unnamed_in_sources(&inv.hypotheses[i], inv) {
                let h = &mut inv.hypotheses[i];
                h.verdict = Verdict::Uncertain;
                h.reasoning = format!("Las fuentes citadas no nombran «{}». {}", names.join("», «"), h.reasoning);
            }
            let h = &inv.hypotheses[i];
            let cites: String = h.supporting.iter().chain(&h.contradicting).map(|n| format!("[{n}]")).collect();
            let message = format!("«{}»: {} ({:.2}) {cites}", h.text, verdict_label(h.verdict), h.confidence);
            if !advance(inv, checkpoint, "Buscando evidencias", &message) {
                return Ok(());
            }
        }

        if inv.resolved() {
            break;
        }
    }

    // ── 5. Conclusión ────────────────────────────────────────────────
    if !advance(inv, checkpoint, "Redactando conclusión", "Redactando la conclusión") {
        return Ok(());
    }
    inv.conclusion = Some(conclude(llm, inv)?);

    // ── 6. Aprendizaje de lo verificado ──────────────────────────────
    let mut learn_from: Vec<usize> = inv.hypotheses
        .iter()
        .filter(|h| h.verdict == Verdict::Supported)
        .flat_map(|h| h.supporting.iter().copied())
        .collect();
    learn_from.sort_unstable();
    learn_from.dedup();
    learn_from.truncate(MAX_LEARNING_SOURCES);
    if !learn_from.is_empty() {
        if !advance(inv, checkpoint, "Aprendiendo", &format!("Aprendiendo de {} fuente(s) verificada(s)", learn_from.len())) {
            return Ok(());
        }
        let hits: Vec<SearchHit> = learn_from
            .iter()
            .filter_map(|n| inv.sources.get(n - 1))
            .map(|s| SearchHit { title: s.title.clone(), url: s.url.clone(), snippet: s.snippet.clone() })
            .collect();
        let learned = research::learn_from_sources(
            llm, search, graph, &inv.objective, &hits, research::LEARNING_THRESHOLD, &|_| {},
        )?;
        inv.learned = learned.consolidated;
    }

    let supported = inv.hypotheses.iter().filter(|h| h.verdict == Verdict::Supported).count();
    inv.finish(InvestigationStatus::Done, "Completa");
    let message = format!(
        "✓ Investigación completa: {supported} de {} hipótesis apoyada(s), {} fuente(s), {} hecho(s) aprendido(s)",
        inv.hypotheses.len(), inv.sources.len(), inv.learned.len()
    );
    checkpoint(inv, &message);
    Ok(())
}

/// Traduce el veredicto del modelo. Sin fuentes citadas que estén entre
/// las que vio, o con certeza por debajo de `MIN_VERDICT_CONFIDENCE`,
/// ningún veredicto cuenta: queda «sin confirmar».
fn apply_judgment(h: &mut Hypothesis, judged: Option<Judgment>, evidence: &[usize]) {
    let Some(j) = judged else {
        h.verdict = Verdict::Uncertain;
        h.reasoning = if evidence.is_empty() { "Sin fuentes para evaluarla".into() } else { "Veredicto no interpretable".into() };
        return;
    };
    let cited = |ns: &[i64]| -> Vec<usize> {
        let mut v: Vec<usize> = ns.iter().filter_map(|&n| usize::try_from(n).ok()).filter(|n| evidence.contains(n)).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    h.supporting = cited(&j.a);
    h.contradicting = cited(&j.x);
    h.confidence = j.c.clamp(0.0, 1.0);
    h.reasoning = j.r.trim().to_string();
    // Fuentes a favor y en contra a la vez: la hipótesis no está resuelta.
    // En vivo, el juez invirtió las citas y «refutó» que Compiere sea un ERP
    // libre citando la propia página de Compiere.
    let mixed = !h.supporting.is_empty() && !h.contradicting.is_empty();
    if mixed {
        h.reasoning = format!("Fuentes en conflicto. {}", h.reasoning);
    }
    let sure = h.confidence >= MIN_VERDICT_CONFIDENCE && !mixed;
    h.verdict = match j.v.trim().to_lowercase().as_str() {
        "apoya" if sure && !h.supporting.is_empty() => Verdict::Supported,
        "contradice" if sure && !h.contradicting.is_empty() => Verdict::Refuted,
        _ => Verdict::Uncertain,
    };
}

/// Palabras en forma canónica (minúsculas, sin tildes), de 3 letras o más.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(canonical_id)
}

/// Palabras con las que suele empezar una frase: con mayúscula inicial
/// solo por ir primero, no por ser nombres propios.
const SENTENCE_STARTERS: &[&str] = &[
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "existen", "existe", "hay", "es", "son",
    "fue", "fueron", "se", "en", "para", "con", "por", "su", "sus", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "ningun", "ninguno", "ninguna", "todos", "todas", "cada", "segun",
    "algunos", "algunas", "muchos", "muchas", "ambos", "otro", "otra", "otros", "otras", "the",
];

/// Si una hipótesis apoyada nombra algo propio que no viene del objetivo
/// (palabras con mayúscula inicial), al menos uno de esos nombres debe
/// aparecer en las fuentes que cita; si no, devuelve los nombres. En vivo,
/// el 2B apoyó «Opkey es un ERP de código abierto» citando una fuente que
/// hablaba de otro producto.
fn unnamed_in_sources(h: &Hypothesis, inv: &Investigation) -> Option<Vec<String>> {
    if h.verdict != Verdict::Supported {
        return None;
    }
    let objective: HashSet<String> = words(&inv.objective).collect();
    let mut names: Vec<String> = Vec::new();
    for word in h.text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 3) {
        let canon = canonical_id(word);
        if word.chars().next().is_some_and(char::is_uppercase)
            && !SENTENCE_STARTERS.contains(&canon.as_str())
            && !objective.contains(&canon)
            && !names.contains(&word.to_string())
        {
            names.push(word.to_string());
        }
    }
    if names.is_empty() {
        return None;
    }
    let cited: HashSet<String> = h.supporting
        .iter()
        .filter_map(|n| inv.sources.get(n - 1))
        .flat_map(|s| words(&s.title).chain(words(&s.snippet)).collect::<Vec<_>>())
        .collect();
    (!names.iter().any(|n| cited.contains(&canonical_id(n)))).then_some(names)
}

/// Conclusión en prosa con las hipótesis evaluadas y las fuentes citadas.
fn conclude<L: LlmBackend>(llm: &L, inv: &Investigation) -> Result<String, BridgeError> {
    let mut cited: Vec<usize> = inv.hypotheses
        .iter()
        .flat_map(|h| h.supporting.iter().chain(&h.contradicting).copied())
        .collect();
    cited.sort_unstable();
    cited.dedup();
    let hypotheses = inv.hypotheses
        .iter()
        .map(|h| {
            let refs: String = h.supporting.iter().chain(&h.contradicting).map(|n| format!(" [{n}]")).collect();
            format!("- {} ({:.2}): «{}»{refs}", verdict_label(h.verdict).to_uppercase(), h.confidence, h.text)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let user = format!(
        "Objetivo: {}\n\nHipótesis evaluadas:\n{}\n\nFuentes:\n{}",
        inv.objective,
        if hypotheses.is_empty() { "(ninguna)".to_string() } else { hypotheses },
        if cited.is_empty() { "(ninguna citada)".to_string() } else { inv.format_sources(&cited) }
    );
    let any_supported = inv.hypotheses.iter().any(|h| h.verdict == Verdict::Supported);
    let raw = llm.complete(prompts::investigation_conclusion_system_prompt(any_supported), &user, CONCLUSION_TOKENS, 0.3)?;
    let text = strip_think(&raw);
    Ok(if text.is_empty() {
        "No se pudo redactar la conclusión; revisa las hipótesis y sus fuentes.".into()
    } else {
        text
    })
}

/// Quita los bloques <think>…</think> de modelos razonadores.
fn strip_think(raw: &str) -> String {
    let mut text = raw.to_string();
    while let (Some(start), Some(end)) = (text.find("<think>"), text.find("</think>")) {
        if end < start {
            break;
        }
        text.replace_range(start..end + "</think>".len(), "");
    }
    text.trim().to_string()
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_graph::MemoryGraph;
    use crate::truth_filter::SearchError;
    use std::sync::{Arc, Mutex};

    /// LLM de guion: responde según el paso (prompt del sistema) y, en el
    /// juicio, según la hipótesis.
    #[derive(Clone, Default)]
    struct ScriptedLlm {
        judgments: Arc<Vec<(&'static str, &'static str)>>,
        calls:     Arc<Mutex<Vec<String>>>,
    }

    impl LlmBackend for ScriptedLlm {
        fn complete(&self, system: &str, user: &str, _: i32, _: f32) -> Result<String, BridgeError> {
            self.calls.lock().unwrap().push(system.chars().take(40).collect());
            let reply = if system.contains("Divide el objetivo") {
                r#"{"q": ["¿Quién descubrió la penicilina?", "¿En qué año se descubrió la penicilina?"]}"#.to_string()
            } else if system.contains("investigación en curso") {
                r#"{"q": ["Alexander Fleming penicilina 1928"]}"#.to_string()
            } else if system.contains("propón de 2 a 4 hipótesis") {
                r#"{"h": ["Alexander Fleming descubrió la penicilina", "Louis Pasteur descubrió la penicilina", "¿Fue Fleming?"]}"#.to_string()
            } else if system.contains("Evalúa una hipótesis") {
                self.judgments
                    .iter()
                    .find(|(h, _)| user.contains(&format!("Hipótesis: «{h}")))
                    .map(|(_, j)| j.to_string())
                    .unwrap_or_else(|| r#"{"v": "insuficiente", "c": 0.3, "a": [], "x": [], "r": "No consta."}"#.into())
            } else if system.contains("Redacta la conclusión") || system.contains("Redacta en 2 a 4") {
                "<think>…</think>La descubrió Alexander Fleming en 1928 [1].".to_string()
            } else if system.contains("Extrae de las fuentes") {
                r#"{"triples": [{"subject": "alexander_fleming", "relation": "descubrio", "object": "penicilina", "weight": 0.9}]}"#.to_string()
            } else {
                // Validador epistémico del aprendizaje.
                r#"{"status": "VERIFIED", "confidence": 0.9, "reasoning": "La fuente lo afirma."}"#.to_string()
            };
            Ok(reply)
        }
    }

    /// Buscador: dos fuentes fijas por consulta, con URL derivada de ella.
    #[derive(Clone, Default)]
    struct FixedSearch {
        down: bool,
    }

    impl SearchBackend for FixedSearch {
        fn search(&self, query: &str) -> Result<Vec<String>, SearchError> {
            Ok(self.search_detailed(query)?.into_iter().map(|h| h.snippet).collect())
        }
        fn search_detailed(&self, query: &str) -> Result<Vec<SearchHit>, SearchError> {
            if self.down {
                return Err(SearchError::Timeout { seconds: 5 });
            }
            Ok(vec![
                SearchHit {
                    title:   "Penicilina - Wikipedia".into(),
                    url:     "https://es.wikipedia.org/wiki/Penicilina".into(),
                    snippet: "Alexander Fleming descubrió la penicilina en 1928.".into(),
                },
                SearchHit {
                    title:   format!("Resultado para {query}"),
                    url:     format!("https://ejemplo.org/{}", query.len()),
                    snippet: "Louis Pasteur estudió los microbios, pero no descubrió la penicilina.".into(),
                },
            ])
        }
    }

    fn llm(judgments: Vec<(&'static str, &'static str)>) -> ScriptedLlm {
        ScriptedLlm { judgments: Arc::new(judgments), ..Default::default() }
    }

    #[test]
    fn test_investigation_resolves_cites_and_learns() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let llm = llm(vec![
            ("Alexander Fleming descubrió", r#"{"v": "apoya", "c": 0.9, "a": [1, 99], "x": [], "r": "La fuente 1 lo afirma."}"#),
            ("Louis Pasteur descubrió", r#"{"v": "contradice", "c": 0.8, "a": [], "x": [2], "r": "La fuente 2 lo niega."}"#),
        ]);
        let mut inv = Investigation::new("¿Quién descubrió la penicilina?", 3);
        let mut messages = Vec::new();
        run(&llm, &FixedSearch::default(), &graph, &mut inv, &[], &mut |_, m: &str| { messages.push(m.to_string()); true }).unwrap();

        assert_eq!(inv.status, InvestigationStatus::Done);
        assert_eq!(inv.round, 1, "resuelta en la primera ronda");
        assert_eq!(inv.hypotheses.len(), 2, "la pregunta «¿Fue Fleming?» no es una hipótesis");
        let fleming = &inv.hypotheses[0];
        assert_eq!((fleming.verdict, fleming.supporting.clone()), (Verdict::Supported, vec![1]), "la cita 99 no estaba entre las fuentes");
        assert_eq!(inv.hypotheses[1].verdict, Verdict::Refuted);
        assert_eq!(inv.conclusion.as_deref(), Some("La descubrió Alexander Fleming en 1928 [1]."));
        assert_eq!(inv.sources[0].n, 1);
        assert_eq!(inv.sources.iter().filter(|s| s.url.contains("wikipedia")).count(), 1, "una URL, una fuente");
        assert_eq!(inv.learned, ["alexander_fleming→descubrio→penicilina"]);
        assert!(messages.last().unwrap().starts_with("✓ Investigación completa"));
    }

    #[test]
    fn test_unresolved_investigation_refines_until_max_rounds() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let llm = llm(vec![]); // el juez nunca encuentra evidencia
        let mut inv = Investigation::new("¿Quién descubrió la penicilina?", 2);
        run(&llm, &FixedSearch::default(), &graph, &mut inv, &[], &mut |_, _: &str| true).unwrap();

        assert_eq!(inv.status, InvestigationStatus::Done);
        assert_eq!(inv.round, 2);
        assert!(inv.questions.contains(&"Alexander Fleming penicilina 1928".to_string()), "la 2.ª ronda reformula");
        assert!(inv.hypotheses.iter().all(|h| h.verdict == Verdict::Uncertain));
        assert!(inv.learned.is_empty(), "sin hipótesis apoyadas no se aprende nada");
    }

    #[test]
    fn test_cancellation_and_search_failure() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let mut inv = Investigation::new("objetivo", 3);
        let mut steps = 0;
        run(&llm(vec![]), &FixedSearch::default(), &graph, &mut inv, &[], &mut |_, _: &str| { steps += 1; steps < 3 }).unwrap();
        assert_eq!(inv.status, InvestigationStatus::Cancelled);
        assert!(inv.finished_at.is_some());

        // Buscador caído: reintenta tras cada espera y luego se rinde.
        let mut inv = Investigation::new("objetivo", 3);
        let mut messages = Vec::new();
        run(&llm(vec![]), &FixedSearch { down: true }, &graph, &mut inv, &[Duration::ZERO; 2], &mut |_, m: &str| { messages.push(m.to_string()); true }).unwrap();
        assert_eq!(inv.status, InvestigationStatus::Failed);
        assert!(inv.error.as_deref().unwrap().contains("falló"));
        assert_eq!(messages.iter().filter(|m| m.contains("reintento")).count(), 2);
    }

    #[test]
    fn test_judgment_requires_cited_evidence() {
        let mut h = Hypothesis {
            text: "x".into(), round: 1, verdict: Verdict::Pending, confidence: 0.0,
            supporting: vec![], contradicting: vec![], reasoning: String::new(),
        };
        let j = |v: &str, a: Vec<i64>| Judgment { v: v.into(), c: 0.95, a, x: vec![], r: "r".into() };
        apply_judgment(&mut h, Some(j("apoya", vec![])), &[1, 2]);
        assert_eq!(h.verdict, Verdict::Uncertain, "apoyar sin citar no cuenta");
        apply_judgment(&mut h, Some(Judgment { c: 0.5, ..j("apoya", vec![1]) }), &[1, 2]);
        assert_eq!(h.verdict, Verdict::Uncertain, "ni con certeza baja");
        apply_judgment(&mut h, Some(Judgment { x: vec![1], ..j("contradice", vec![2]) }), &[1, 2]);
        assert_eq!(h.verdict, Verdict::Uncertain, "con fuentes en conflicto no se resuelve");
        assert!(h.reasoning.starts_with("Fuentes en conflicto"));
        apply_judgment(&mut h, Some(j("apoya", vec![2, 2, 7])), &[1, 2]);
        assert_eq!((h.verdict, h.supporting.clone()), (Verdict::Supported, vec![2]));
        apply_judgment(&mut h, None, &[1]);
        assert_eq!(h.verdict, Verdict::Uncertain);

        // Anclaje: los nombres propios nuevos deben estar en las fuentes citadas.
        let mut inv = Investigation::new("¿Qué ERP de código abierto existen para empresas?", 1);
        inv.add_sources(&[SearchHit { title: "Compiere".into(), url: "u1".into(), snippet: "Compiere es un ERP libre.".into() }], 1);
        let supported = |text: &str| Hypothesis {
            text: text.into(), round: 1, verdict: Verdict::Supported, confidence: 0.8,
            supporting: vec![1], contradicting: vec![], reasoning: String::new(),
        };
        assert_eq!(unnamed_in_sources(&supported("Compiere es un ERP de código abierto"), &inv), None);
        assert_eq!(unnamed_in_sources(&supported("Opkey es un ERP de código abierto para empresas"), &inv),
            Some(vec!["Opkey".to_string()]), "el caso real de la prueba en vivo");
        assert_eq!(unnamed_in_sources(&supported("Existen ERP como Compiere y Opkey"), &inv), None, "basta con uno");
        assert_eq!(unnamed_in_sources(&supported("Existen tres ERP libres para pymes"), &inv), None, "sin nombres propios");
        assert_eq!(strip_think("<think>a</think> hola <think>b</think>"), "hola");
    }
}
