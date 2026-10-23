// ============================================================
// working_memory.rs — Lóbulo Frontal: Memoria de Trabajo (Paso 13)
//
// RESPONSABILIDAD:
//   Gestionar el contexto de conversación en RAM (corto plazo).
//   Cada `session_id` mantiene una ventana deslizante de los
//   últimos `WINDOW_SIZE` mensajes (usuario + asistente).
//
// DECISIONES DE DISEÑO:
//   • `RwLock<HashMap<...>>`:
//       Múltiples lectores concurrentes (Axum tiene N hilos) sin bloqueo.
//       Escritura (push/trim) requiere lock exclusivo durante microsegundos.
//       Alternativa `dashmap` añade dependencia; `RwLock` std es suficiente
//       para el patrón de acceso de una API conversacional (<100 sesiones).
//   • Ventana deslizante de `WINDOW_SIZE` mensajes:
//       Limita el crecimiento de RAM a O(WINDOW_SIZE * avg_message_len).
//       En un chat de texto (<500 chars/mensaje, 6 mensajes) ≈ 3 KB/sesión.
//   • `Arc<SessionManager>` en AppState:
//       Axum clona el estado por handler; el Arc garantiza que todos
//       los handlers operan sobre el mismo mapa, no copias.
// ============================================================

use std::collections::HashMap;
use std::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Número máximo de mensajes (turnos) que se retienen por sesión.
/// 6 mensajes = 3 turnos: Usuario-Asistente × 3.
/// Suficiente para resolver correferencias en la mayoría de conversaciones.
const WINDOW_SIZE: usize = 6;

// ──────────────────────────────────────────────────────────────
//  Message — Unidad de Historia de Conversación
// ──────────────────────────────────────────────────────────────

/// Un mensaje en la historia de conversación.
///
/// `role` sigue la convención estándar de LLMs instruction-tuned:
///   - `"user"`:      Turno del usuario.
///   - `"assistant"`: Turno del agente.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role:    String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }
}

// ──────────────────────────────────────────────────────────────
//  SessionManager — Registro central de sesiones
// ──────────────────────────────────────────────────────────────

/// Gestor de memoria de trabajo para sesiones de conversación.
///
/// Thread-safe via `RwLock<HashMap<...>>`. Diseñado para vivir en
/// `AppState` envuelto en `Arc<SessionManager>`.
pub struct SessionManager {
    /// Mapa de `session_id` → historial de mensajes (ventana deslizante).
    sessions: RwLock<HashMap<String, Vec<Message>>>,
}

impl SessionManager {
    /// Crea un nuevo `SessionManager` vacío.
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Añade un mensaje al historial de la sesión indicada.
    ///
    /// Si la sesión no existe, la crea automáticamente.
    /// Si el historial supera `WINDOW_SIZE`, descarta los mensajes más
    /// antiguos manteniendo los `WINDOW_SIZE` más recientes (sliding window).
    ///
    /// # Concurrencia
    /// Adquiere un write-lock brevemente (~microsegundos). No bloquea
    /// a los handlers que solo leen (`get_history`).
    pub fn push(&self, session_id: &str, message: Message) {
        let mut map = self.sessions.write().unwrap_or_else(|e| e.into_inner());
        let history = map.entry(session_id.to_string()).or_default();
        history.push(message);

        // Ventana deslizante: mantener solo los WINDOW_SIZE últimos mensajes.
        if history.len() > WINDOW_SIZE {
            let drain_count = history.len() - WINDOW_SIZE;
            history.drain(..drain_count);
        }
    }

    /// Devuelve el historial de la sesión (clon para evitar mantener el lock).
    ///
    /// Retorna `Vec` vacío si la sesión no existe todavía.
    ///
    /// # Concurrencia
    /// Adquiere un read-lock compartido: múltiples handlers pueden leer
    /// simultáneamente sin bloquearse entre sí.
    pub fn get_history(&self, session_id: &str) -> Vec<Message> {
        let map = self.sessions.read().unwrap_or_else(|e| e.into_inner());
        map.get(session_id).cloned().unwrap_or_default()
    }

    /// Retorna el número de sesiones activas (útil para métricas/tests).
    pub fn session_count(&self) -> usize {
        let map = self.sessions.read().unwrap_or_else(|e| e.into_inner());
        map.len()
    }

    /// Limpia todas las sesiones (útil en tests o ante un reinicio).
    #[cfg(test)]
    pub fn clear(&self) {
        let mut map = self.sessions.write().unwrap_or_else(|e| e.into_inner());
        map.clear();
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────
//  Helper: formatear historial como texto de contexto
// ──────────────────────────────────────────────────────────────

/// Serializa el historial de una sesión en texto plano para inyectar
/// en el prompt de correferencia del LLM.
///
/// Formato:
/// ```text
/// User: ¿Qué es Fénix Flow?
/// Assistant: Es un software de facturación.
/// ```
///
/// La representación en prosa es más eficiente en tokens que JSON
/// para modelos instruction-tuned pequeños (<2B parámetros).
pub fn format_history_as_text(history: &[Message]) -> String {
    if history.is_empty() {
        return String::new();
    }
    history
        .iter()
        .map(|m| {
            let role_label = match m.role.as_str() {
                "user"      => "User",
                "assistant" => "Assistant",
                other       => other,
            };
            format!("{}: {}", role_label, m.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ──────────────────────────────────────────────────────────────
//  TESTS
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sm() -> SessionManager { SessionManager::new() }

    #[test]
    fn test_push_creates_session_on_first_message() {
        let sm = make_sm();
        sm.push("sess1", Message::user("Hola"));
        assert_eq!(sm.session_count(), 1);
        let h = sm.get_history("sess1");
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].role, "user");
        assert_eq!(h[0].content, "Hola");
    }

    #[test]
    fn test_get_history_unknown_session_returns_empty() {
        let sm = make_sm();
        let h = sm.get_history("nonexistent");
        assert!(h.is_empty(), "Sesión inexistente debe retornar Vec vacío");
    }

    #[test]
    fn test_sliding_window_trims_oldest_messages() {
        let sm = make_sm();
        // Insertar WINDOW_SIZE + 2 mensajes.
        for i in 0..(WINDOW_SIZE + 2) {
            sm.push("s", Message::user(format!("msg_{}", i)));
        }
        let h = sm.get_history("s");
        assert_eq!(h.len(), WINDOW_SIZE, "La ventana debe limitar a WINDOW_SIZE");
        // Los primeros 2 mensajes deben haber sido descartados.
        assert_eq!(h[0].content, format!("msg_{}", 2));
        assert_eq!(h[WINDOW_SIZE - 1].content, format!("msg_{}", WINDOW_SIZE + 1));
    }

    #[test]
    fn test_push_exactly_window_size_messages_no_trim() {
        let sm = make_sm();
        for i in 0..WINDOW_SIZE {
            sm.push("s", Message::user(format!("m{}", i)));
        }
        assert_eq!(sm.get_history("s").len(), WINDOW_SIZE);
    }

    #[test]
    fn test_multiple_sessions_are_independent() {
        let sm = make_sm();
        sm.push("alice", Message::user("¿Qué es Rust?"));
        sm.push("alice", Message::assistant("Un lenguaje de sistemas."));
        sm.push("bob",   Message::user("¿Qué es Python?"));

        assert_eq!(sm.get_history("alice").len(), 2);
        assert_eq!(sm.get_history("bob").len(),   1);
        assert_eq!(sm.session_count(), 2);
    }

    #[test]
    fn test_message_constructors() {
        let u = Message::user("hello");
        let a = Message::assistant("world");
        assert_eq!(u.role, "user");
        assert_eq!(a.role, "assistant");
    }

    #[test]
    fn test_format_history_empty_returns_empty_string() {
        assert_eq!(format_history_as_text(&[]), "");
    }

    #[test]
    fn test_format_history_renders_correctly() {
        let history = vec![
            Message::user("¿Qué es Fénix Flow?"),
            Message::assistant("Es un software de facturación."),
        ];
        let text = format_history_as_text(&history);
        assert!(text.contains("User: ¿Qué es Fénix Flow?"));
        assert!(text.contains("Assistant: Es un software de facturación."));
    }

    #[test]
    fn test_session_manager_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SessionManager>();
    }

    #[test]
    fn test_concurrent_push_does_not_panic() {
        use std::sync::Arc;
        use std::thread;

        let sm = Arc::new(SessionManager::new());
        let mut handles = vec![];
        for i in 0..10 {
            let sm_clone = Arc::clone(&sm);
            let session = format!("session_{}", i % 3); // 3 sesiones concurrentes
            handles.push(thread::spawn(move || {
                for j in 0..10 {
                    sm_clone.push(&session, Message::user(format!("msg_{}_{}", i, j)));
                }
            }));
        }
        for h in handles { h.join().unwrap(); }

        // Sin panic y con datos coherentes.
        for i in 0..3 {
            let session = format!("session_{}", i);
            let hist = sm.get_history(&session);
            assert!(hist.len() <= WINDOW_SIZE, "Ventana deslizante debe respetarse");
        }
    }
}
