// ============================================================
// priority.rs — Prioridad Cognitiva (Paso 21)
//
// PROBLEMA:
//   Ollama atiende las inferencias en serie (o reparte la GPU entre ellas).
//   El subconsciente encadena 3-4 inferencias por ciclo cada 3 s, así que
//   cada mensaje del usuario esperaba detrás de un ciclo completo.
//
// DISEÑO:
//   • Cada handler de usuario (chat, query, learn) toma un `ForegroundGuard`
//     (RAII) mientras dura su trabajo, incluido el streaming SSE entero.
//   • El subconsciente envuelve su LLM en `BackgroundLlm`: antes de CADA
//     inferencia espera a que no haya trabajo en primer plano y haya pasado
//     un periodo de gracia (el usuario suele encadenar mensajes).
//   • El punto de cesión es cada llamada al LLM, no cada ciclo: un ciclo a
//     medias se pausa entre pasos en lugar de terminar entero.
//
// LÍMITE CONOCIDO:
//   Una inferencia del subconsciente que ya está en curso no se interrumpe;
//   el chat espera, como mucho, a que termine esa única llamada.
//   Con uso continuo del chat el subconsciente no avanza: es intencional.
// ============================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::llm_bridge::{BoxedTokenStream, BridgeError, LlmBackend};

/// Tiempo que el subconsciente sigue en pausa tras terminar la última
/// petición del usuario.
pub const DEFAULT_GRACE: Duration = Duration::from_millis(1500);

/// Cadencia de comprobación mientras hay trabajo en primer plano.
/// Despreciable frente a la latencia de una inferencia.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Centinela: todavía no ha terminado ninguna petición en primer plano.
const NEVER: u64 = u64::MAX;

/// Árbitro entre el trabajo en primer plano (usuario) y el de fondo
/// (subconsciente). Compartido vía `Arc` entre `AppState` y el bucle de fondo.
pub struct CognitivePriority {
    active:          AtomicUsize,
    /// Milisegundos desde `epoch` en que terminó la última petición del usuario.
    last_release_ms: AtomicU64,
    epoch:           Instant,
    grace:           Duration,
}

impl CognitivePriority {
    pub fn new(grace: Duration) -> Self {
        Self {
            active:          AtomicUsize::new(0),
            last_release_ms: AtomicU64::new(NEVER),
            epoch:           Instant::now(),
            grace,
        }
    }

    /// Marca el inicio de trabajo en primer plano. La prioridad dura lo
    /// que viva el guard devuelto.
    pub fn enter(self: &Arc<Self>) -> ForegroundGuard {
        self.active.fetch_add(1, Ordering::SeqCst);
        ForegroundGuard(Arc::clone(self))
    }

    /// Peticiones del usuario en curso.
    pub fn foreground_active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// `None` si el trabajo de fondo puede continuar; `Some(d)` si debe
    /// esperar al menos `d` antes de volver a comprobar.
    fn must_wait(&self) -> Option<Duration> {
        if self.active.load(Ordering::SeqCst) > 0 {
            return Some(POLL_INTERVAL);
        }
        let last = self.last_release_ms.load(Ordering::SeqCst);
        if last == NEVER {
            return None;
        }
        let since = self.now_ms().saturating_sub(last);
        let grace = self.grace.as_millis() as u64;
        (since < grace).then(|| Duration::from_millis(grace - since))
    }

    /// Espera bloqueante (hilos de `spawn_blocking`). Devuelve el tiempo esperado.
    pub fn wait_idle_blocking(&self) -> Duration {
        let start = Instant::now();
        while let Some(d) = self.must_wait() {
            std::thread::sleep(d);
        }
        start.elapsed()
    }

    /// Espera asíncrona (tareas de Tokio). Devuelve el tiempo esperado.
    pub async fn wait_idle(&self) -> Duration {
        let start = Instant::now();
        while let Some(d) = self.must_wait() {
            tokio::time::sleep(d).await;
        }
        start.elapsed()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }
}

impl Default for CognitivePriority {
    fn default() -> Self {
        Self::new(DEFAULT_GRACE)
    }
}

/// Prioridad del usuario activa mientras viva. Se libera en `Drop`, también
/// si el cliente corta el stream SSE o el handler devuelve un error.
pub struct ForegroundGuard(Arc<CognitivePriority>);

impl Drop for ForegroundGuard {
    fn drop(&mut self) {
        // Primero la marca de tiempo, luego el decremento: quien vea
        // `active == 0` verá también el periodo de gracia actualizado.
        self.0.last_release_ms.store(self.0.now_ms(), Ordering::SeqCst);
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Transición observable de una cesión de turno (para telemetría).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum YieldEvent {
    /// El trabajo de fondo se detiene porque el usuario tiene prioridad.
    Paused,
    /// El trabajo de fondo se reanuda tras esperar este tiempo.
    Resumed(Duration),
}

/// Backend LLM para trabajo de fondo: cede el turno al usuario antes de
/// cada inferencia. Envuelve cualquier `LlmBackend`.
pub struct BackgroundLlm<L> {
    inner:    L,
    priority: Arc<CognitivePriority>,
    on_yield: Option<Box<dyn Fn(YieldEvent) + Send + Sync>>,
}

impl<L: LlmBackend> BackgroundLlm<L> {
    pub fn new(inner: L, priority: Arc<CognitivePriority>) -> Self {
        Self { inner, priority, on_yield: None }
    }

    /// Callback opcional para emitir la pausa/reanudación a la telemetría.
    pub fn on_yield(mut self, f: impl Fn(YieldEvent) + Send + Sync + 'static) -> Self {
        self.on_yield = Some(Box::new(f));
        self
    }

    fn yield_to_foreground(&self) {
        if self.priority.must_wait().is_none() {
            return;
        }
        let notify = |e| if let Some(f) = &self.on_yield { f(e) };
        notify(YieldEvent::Paused);
        let waited = self.priority.wait_idle_blocking();
        notify(YieldEvent::Resumed(waited));
    }
}

impl<L: LlmBackend> LlmBackend for BackgroundLlm<L> {
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        self.yield_to_foreground();
        self.inner.complete(system_prompt, user_prompt, max_tokens, temperature)
    }

    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        self.yield_to_foreground();
        self.inner.complete_json(system_prompt, user_prompt, schema, max_tokens, temperature)
    }

    fn stream_synthesis(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<BoxedTokenStream, BridgeError> {
        self.yield_to_foreground();
        self.inner.stream_synthesis(system_prompt, user_prompt, max_tokens, temperature)
    }

    fn scope(&self) -> crate::brains::Scope {
        self.inner.scope()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  TESTS
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_bridge::mock::MockLlmBackend;
    use std::sync::Mutex;

    #[test]
    fn test_idle_by_default() {
        let p = CognitivePriority::new(Duration::from_millis(200));
        assert_eq!(p.foreground_active(), 0);
        assert!(p.must_wait().is_none(), "sin actividad previa no hay que esperar");
    }

    #[test]
    fn test_guard_counts_and_releases() {
        let p = Arc::new(CognitivePriority::new(Duration::ZERO));
        let g1 = p.enter();
        let g2 = p.enter();
        assert_eq!(p.foreground_active(), 2);
        drop(g1);
        assert_eq!(p.foreground_active(), 1);
        drop(g2);
        assert_eq!(p.foreground_active(), 0);
        assert!(p.must_wait().is_none());
    }

    #[test]
    fn test_grace_period_after_release() {
        let p = Arc::new(CognitivePriority::new(Duration::from_millis(150)));
        drop(p.enter());
        let w = p.must_wait().expect("recién liberado: debe esperar la gracia");
        assert!(w <= Duration::from_millis(150));
        let waited = p.wait_idle_blocking();
        assert!(waited >= Duration::from_millis(100), "esperó {waited:?}");
        assert!(p.must_wait().is_none());
    }

    #[test]
    fn test_background_llm_waits_for_foreground() {
        let p = Arc::new(CognitivePriority::new(Duration::ZERO));
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = Arc::clone(&events);
        let llm = BackgroundLlm::new(MockLlmBackend::new(vec!["ok"]), Arc::clone(&p))
            .on_yield(move |e| ev.lock().unwrap().push(e));

        let guard = p.enter();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            drop(guard);
        });

        let t0 = Instant::now();
        let out = llm.complete("s", "u", 8, 0.0).unwrap();
        releaser.join().unwrap();

        assert_eq!(out, "ok");
        assert!(t0.elapsed() >= Duration::from_millis(100), "la inferencia debió esperar al usuario");
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], YieldEvent::Paused);
        assert!(matches!(events[1], YieldEvent::Resumed(d) if d >= Duration::from_millis(100)));
    }

    #[test]
    fn test_background_llm_no_events_when_idle() {
        let p = Arc::new(CognitivePriority::new(Duration::ZERO));
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = Arc::clone(&events);
        let llm = BackgroundLlm::new(MockLlmBackend::new(vec!["ok"]), p)
            .on_yield(move |e| ev.lock().unwrap().push(e));

        assert_eq!(llm.complete("s", "u", 8, 0.0).unwrap(), "ok");
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_async_wait_idle() {
        let p = Arc::new(CognitivePriority::new(Duration::ZERO));
        let guard = p.enter();
        let p2 = Arc::clone(&p);
        let waiter = tokio::spawn(async move { p2.wait_idle().await });
        tokio::time::sleep(Duration::from_millis(80)).await;
        drop(guard);
        let waited = waiter.await.unwrap();
        assert!(waited >= Duration::from_millis(60), "esperó {waited:?}");
    }
}
