//! **Modo pensar** (Paso 33): ideas propias que conectan lo que sabe.
//!
//! «Piensa…», «imagina…», «crea un sistema…» no piden un dato sino
//! construir algo. El cerebro profundo lo hace en tres pasos encadenados,
//! cada uno sobre el anterior: propone ideas a partir del material (la
//! memoria y, si no sabe nada del tema, una búsqueda rápida), las conecta en
//! un todo y construye el resultado. El alcance del cerebro decide cuántas
//! ideas y cuánto texto: un 2B no se ve aplastado y un 35B llega más lejos.
//!
//! Las ideas se guardan como pasajes de origen `Inferred` con el título
//! «Ideas propias (sin verificar)»: la memoria las recuerda como lo que son.

use crate::brains::Scope;
use crate::llm_bridge::{BridgeError, LlmBackend};
use crate::memory_graph::{drafts_from_text, save_passages, GraphAccess};
use crate::prompts;
use crate::types::EdgeOrigin;

/// Temperatura de los pasos: más alta que al responder, para que haya ideas.
pub const THINK_TEMPERATURE: f32 = 0.7;

/// Lo que llega al historial de la sesión: el resultado entero haría enorme
/// el prompt del planificador en el siguiente turno.
const HISTORY_CHARS: usize = 1500;

/// Comienzo del título de las ideas guardadas.
pub const IDEAS_TITLE: &str = "Ideas propias (sin verificar)";

/// Caracteres del objetivo en ese título.
const TITLE_OBJECTIVE_CHARS: usize = 80;

/// Un paso del modo pensar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Ideas,
    Connections,
    Result,
}

impl Step {
    pub const ALL: [Step; 3] = [Step::Ideas, Step::Connections, Step::Result];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ideas       => "ideas",
            Self::Connections => "conexiones",
            Self::Result      => "resultado",
        }
    }

    /// Aviso mientras el cerebro empieza a escribir el paso.
    pub fn status(self) -> &'static str {
        match self {
            Self::Ideas       => "Proponiendo ideas…",
            Self::Connections => "Conectando las ideas…",
            Self::Result      => "Construyendo el resultado…",
        }
    }

    /// Encabezado del paso en el chat.
    pub fn heading(self) -> &'static str {
        match self {
            Self::Ideas       => "### 1 · Ideas\n\n",
            Self::Connections => "\n\n### 2 · Cómo se conectan\n\n",
            Self::Result      => "\n\n### 3 · Resultado\n\n",
        }
    }

    pub fn max_tokens(self, scope: &Scope) -> i32 {
        match self {
            Self::Result => scope.think_tokens,
            _ => scope.think_step_tokens,
        }
    }

    /// Prompt del paso: el objetivo, el material y los pasos ya escritos.
    pub fn user_prompt(self, objective: &str, material: &str, done: &[String], scope: &Scope) -> String {
        let step = |i: usize| done.get(i).map(String::as_str).unwrap_or("");
        match self {
            Self::Ideas       => prompts::think_ideas_prompt(objective, material, scope.think_ideas),
            Self::Connections => prompts::think_connections_prompt(objective, material, step(0)),
            Self::Result      => prompts::think_result_prompt(objective, material, step(0), step(1)),
        }
    }
}

/// Lo pensado: el texto de cada paso, en orden.
#[derive(Debug, Clone, Default)]
pub struct Thought {
    pub steps: Vec<String>,
}

impl Thought {
    /// Texto completo con los encabezados, como se ve en el chat.
    pub fn text(&self) -> String {
        Step::ALL.iter().zip(&self.steps).map(|(step, text)| format!("{}{}", step.heading(), text.trim())).collect()
    }
}

/// Los tres pasos sin streaming (`POST /query`).
pub fn run(llm: &dyn LlmBackend, objective: &str, material: &str) -> Result<Thought, BridgeError> {
    let scope = llm.scope();
    let mut thought = Thought::default();
    for step in Step::ALL {
        let text = llm.complete(
            prompts::think_system_prompt(),
            &step.user_prompt(objective, material, &thought.steps, &scope),
            step.max_tokens(&scope),
            THINK_TEMPERATURE,
        )?;
        thought.steps.push(text);
    }
    Ok(thought)
}

/// Guarda las ideas como propias y sin verificar. Devuelve cuántos pasajes.
pub fn save_ideas<G: GraphAccess>(graph: &G, objective: &str, ideas: &str) -> usize {
    let objective = shorten(objective.trim(), TITLE_OBJECTIVE_CHARS);
    let title = format!("{IDEAS_TITLE}: {objective}");
    let drafts = drafts_from_text(&without_marks(ideas), EdgeOrigin::Inferred)
        .into_iter()
        .map(|mut draft| {
            draft.title = title.clone();
            draft.question = Some(objective.clone());
            draft
        })
        .collect();
    save_passages(graph, drafts).unwrap_or_else(|e| {
        log::warn!("No se pudieron guardar las ideas: {e}");
        0
    })
}

/// Quita las marcas de certeza de la memoria («[certeza: establecido, peso:
/// 0.90]»): el modelo las copiaba en sus ideas aunque se le pidiera que no.
pub fn without_marks(text: &str) -> String {
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[certeza:") {
        clean.push_str(rest[..start].trim_end_matches(' '));
        rest = rest[start..].find(']').map_or("", |end| &rest[start + end + 1..]);
    }
    clean.push_str(rest);
    clean
}

/// Corta en el último espacio antes de `max` caracteres, con «…».
fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    let head = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{}…", head.trim_end_matches([',', ';', ':', ' ']))
}

/// Lo pensado tal como entra en el historial de la sesión.
pub fn for_history(text: &str) -> String {
    if text.chars().count() <= HISTORY_CHARS {
        return text.to_string();
    }
    let cut: String = text.chars().take(HISTORY_CHARS).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_graph::MemoryGraph;
    use std::sync::Mutex;

    /// Responde según el paso y anota el prompt y los tokens de cada uno.
    struct Steps {
        scope: Scope,
        calls: Mutex<Vec<(String, i32)>>,
    }

    impl LlmBackend for Steps {
        fn complete(&self, _system: &str, user: &str, max_tokens: i32, _t: f32) -> Result<String, BridgeError> {
            self.calls.lock().unwrap().push((user.to_string(), max_tokens));
            Ok(match self.calls.lock().unwrap().len() {
                1 => "1. **Agenda viva**: tareas que se reordenan solas.\n2. **Memoria de hábitos**: aprende tus horarios.".into(),
                2 => "La memoria de hábitos alimenta a la agenda viva.".into(),
                _ => "Un planificador en Python con dos módulos.".into(),
            })
        }

        fn scope(&self) -> Scope {
            self.scope
        }
    }

    #[test]
    fn test_three_steps_build_on_each_other_with_the_brain_scope() {
        let llm = Steps { scope: crate::brains::Tier::Avanzado.scope(), calls: Mutex::new(Vec::new()) };
        let thought = run(&llm, "crear un planificador en Python", "Memoria:\n(nada sobre esto)").unwrap();
        let calls = llm.calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert!(calls[0].0.contains("Propón 5 ideas"), "el avanzado propone 5 ideas: {}", calls[0].0);
        assert!(calls[1].0.contains("Agenda viva"), "las conexiones parten de las ideas");
        assert!(calls[2].0.contains("Agenda viva") && calls[2].0.contains("alimenta a la agenda"));
        assert_eq!(calls[2].1, llm.scope.think_tokens, "el resultado usa el alcance del cerebro");
        assert_eq!(calls[0].1, llm.scope.think_step_tokens);
        let text = thought.text();
        assert!(text.starts_with("### 1 · Ideas"));
        assert!(text.contains("### 3 · Resultado\n\nUn planificador"));
    }

    #[test]
    fn test_ideas_are_saved_as_own_and_unverified() {
        let graph = Mutex::new(MemoryGraph::open(":memory:").unwrap());
        let ideas = "1. **Agenda viva**: tareas que se reordenan solas según la energía del día.";
        assert_eq!(save_ideas(&graph, "crear un planificador", ideas), 1);
        let g = graph.lock().unwrap();
        let (passage, _) = g.relevant_passages("agenda viva", None, 3).unwrap().remove(0);
        assert_eq!(passage.origin, EdgeOrigin::Inferred);
        assert_eq!(passage.display_title(), "Ideas propias (sin verificar): crear un planificador");
    }

    #[test]
    fn test_marks_and_long_objectives_are_trimmed() {
        assert_eq!(
            without_marks("- python es lenguaje [certeza: establecido, peso: 0.90]\nmás [certeza: establecido]."),
            "- python es lenguaje\nmás."
        );
        let objective = "Crear un sistema avanzado en Python que integre la investigación histórica del lenguaje y utilidades";
        assert_eq!(shorten(objective, 80), "Crear un sistema avanzado en Python que integre la investigación histórica del…");
    }

    #[test]
    fn test_history_gets_a_shortened_copy() {
        assert_eq!(for_history("corto"), "corto");
        let long = "a".repeat(HISTORY_CHARS + 10);
        assert_eq!(for_history(&long).chars().count(), HISTORY_CHARS + 1);
    }
}
