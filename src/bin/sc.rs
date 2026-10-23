// ============================================================
// sc.rs — SynaptoCore CLI Periférica (Paso 11)
//
// Binario compañero que actúa como thin-client del servidor
// SynaptoCore (Paso 7). Toda la lógica cognitiva permanece en
// el servidor; este binario solo serializa, envía y formatea.
//
// COMANDOS:
//   sc query "¿pregunta?"          → POST /api/v1/query
//   sc learn "texto"               → POST /api/v1/learn
//   sc learn --file ruta.txt       → POST /api/v1/learn (desde archivo)
//   sc export [--out snap.synapse] → POST /api/v1/federation/export
//   sc watch                       → WS  /ws/subconscious (stream ANSI)
//   sc health                      → GET /health
//
// CONFIGURACIÓN:
//   SC_HOST=http://1.2.3.4:8080   sobrescribe el host base (default: 127.0.0.1:8080)
//
// DISEÑO:
//   • reqwest::Client async       — peticiones HTTP
//   • tokio-tungstenite           — cliente WebSocket para `watch`
//   • clap derive                 — parsing de args sin boilerplate
//   • colored                     — ANSI multiplataforma (Windows 10+)
//   • Manejo de errores UX:        mensajes limpios en rojo, sin panics
// ============================================================

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use colored::Colorize;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_tungstenite::{connect_async, tungstenite::Message};

// ──────────────────────────────────────────────────────────────
//  CLI — definición de comandos con clap derive
// ──────────────────────────────────────────────────────────────

/// sc — SynaptoCore CLI
///
/// Controla e inyecta conocimiento en el cerebro SynaptoCore desde
/// la terminal. El servidor debe estar corriendo en SC_HOST
/// (default: http://127.0.0.1:8080).
#[derive(Parser)]
#[command(
    name        = "sc",
    version     = "0.11.0",
    author      = "SynaptoCore Team",
    about       = "CLI periférica del cerebro cognitivo SynaptoCore",
    long_about  = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Consulta al agente con una pregunta en lenguaje natural.
    ///
    /// Ejemplo: sc query "¿Cuál es la relación entre Rust y WebAssembly?"
    Query {
        /// La pregunta a responder.
        prompt: String,
    },

    /// Inyecta texto nuevo en la memoria del agente.
    ///
    /// Ejemplos:
    ///   sc learn "Rust fue creado por Graydon Hoare en Mozilla en 2010."
    ///   sc learn --file articulo.txt
    Learn {
        /// Texto a memorizar (alternativo a --file).
        text: Option<String>,

        /// Ruta a un archivo de texto a memorizar.
        #[arg(short, long, value_name = "ARCHIVO")]
        file: Option<PathBuf>,
    },

    /// Exporta un delta federado (.synapse) de las últimas 24 horas.
    ///
    /// Ejemplo: sc export --out snapshot.synapse
    Export {
        /// Archivo de salida. Si se omite, escribe en stdout.
        #[arg(short, long, value_name = "ARCHIVO")]
        out: Option<PathBuf>,
    },

    /// Conecta al feed de telemetría en tiempo real del subconsciente.
    ///
    /// Muestra cada evento del CuriosityEngine con colores por tipo.
    /// Ctrl+C para salir.
    Watch,

    /// Muestra el estado del cerebro (nodos, aristas, uptime).
    Health,
}

// ──────────────────────────────────────────────────────────────
//  DTOs de respuesta de la API
// ──────────────────────────────────────────────────────────────

/// Respuesta de GET /health
#[derive(Deserialize)]
struct HealthResponse {
    status:      String,
    total_nodes: u64,
    total_edges: u64,
}

/// Respuesta de POST /api/v1/query
#[derive(Deserialize)]
struct QueryResponse {
    response: String,
}

/// Respuesta de POST /api/v1/learn
#[derive(Deserialize)]
struct LearnResponse {
    triples_extracted: usize,
    llm_attempts:      usize,
}

/// Evento de WebSocket del subconsciente (SubconsciousEvent en el servidor)
#[derive(Deserialize)]
struct SubconsciousEvent {
    event_type: String,
    message:    String,
    timestamp:  u64,
}

// ──────────────────────────────────────────────────────────────
//  Punto de entrada
// ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Habilitar soporte de colores ANSI en Windows 10+ (si no están ya activos).
    #[cfg(windows)]
    colored::control::set_virtual_terminal(true).ok();

    let cli = Cli::parse();

    // Host base: variable de entorno SC_HOST o 127.0.0.1:8080 por defecto.
    let host = std::env::var("SC_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    // Eliminar trailing slash para concatenar rutas limpiamente.
    let host = host.trim_end_matches('/').to_string();

    // Cliente HTTP reutilizable (pool interno de conexiones).
    let client = Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("No se pudo construir el cliente HTTP");

    // Despachar al subcomando correspondiente.
    let result = match cli.command {
        Commands::Query  { prompt }       => cmd_query(&client, &host, &prompt).await,
        Commands::Learn  { text, file }   => cmd_learn(&client, &host, text, file).await,
        Commands::Export { out }          => cmd_export(&client, &host, out).await,
        Commands::Watch                   => cmd_watch(&host).await,
        Commands::Health                  => cmd_health(&client, &host).await,
    };

    // Manejo central de errores: error limpio en rojo, sin panic.
    if let Err(e) = result {
        eprintln!("{}", e.red().bold());
        std::process::exit(1);
    }
}

// ──────────────────────────────────────────────────────────────
//  sc query
// ──────────────────────────────────────────────────────────────

async fn cmd_query(
    client: &Client,
    host:   &str,
    prompt: &str,
) -> Result<(), String> {
    println!("{} {}", "◆ Consultando:".cyan().bold(), prompt.white());

    let t0  = Instant::now();
    let url = format!("{}/api/v1/query", host);

    let resp = client
        .post(&url)
        .json(&json!({ "prompt": prompt }))
        .send()
        .await
        .map_err(|e| format_reqwest_error(e, host))?;

    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().await.unwrap_or_default();
        let msg = body["error"].as_str().unwrap_or("error desconocido");
        return Err(format!("Error {}: {}", status.as_u16(), msg));
    }

    let data: QueryResponse = resp.json().await.map_err(|e| e.to_string())?;
    let latency = t0.elapsed();

    println!();
    println!("{}", "─".repeat(60).dimmed());
    // Preservar saltos de línea en la respuesta del agente.
    for line in data.response.lines() {
        println!("  {}", line.white());
    }
    println!("{}", "─".repeat(60).dimmed());
    println!(
        "{} {}ms",
        "⏱ Latencia:".dimmed(),
        latency.as_millis().to_string().yellow()
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────
//  sc learn
// ──────────────────────────────────────────────────────────────

async fn cmd_learn(
    client: &Client,
    host:   &str,
    text:   Option<String>,
    file:   Option<PathBuf>,
) -> Result<(), String> {
    // Resolver el texto: argumento posicional o archivo.
    let content = match (text, file) {
        (Some(t), _) => t,
        (None, Some(path)) => {
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| format!("No se pudo leer '{}': {}", path.display(), e))?
        }
        (None, None) => {
            return Err(
                "Debes proveer texto o usar --file <ARCHIVO>. \
                 Ejemplo: sc learn \"texto\" o sc learn --file doc.txt"
                    .to_string(),
            );
        }
    };

    if content.trim().is_empty() {
        return Err("El texto está vacío. Nada que aprender.".to_string());
    }

    println!(
        "{} {} caracteres…",
        "◆ Memorizando:".cyan().bold(),
        content.len().to_string().yellow()
    );

    let t0  = Instant::now();
    let url = format!("{}/api/v1/learn", host);

    let resp = client
        .post(&url)
        .json(&json!({ "text": content }))
        .send()
        .await
        .map_err(|e| format_reqwest_error(e, host))?;

    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().await.unwrap_or_default();
        let msg = body["error"].as_str().unwrap_or("error desconocido");
        return Err(format!("Error {}: {}", status.as_u16(), msg));
    }

    let data: LearnResponse = resp.json().await.map_err(|e| e.to_string())?;
    let latency = t0.elapsed();

    println!(
        "{} {} tripletas extraídas en {} intento(s) LLM {}",
        "✓".green().bold(),
        data.triples_extracted.to_string().green().bold(),
        data.llm_attempts.to_string().yellow(),
        format!("({}ms)", latency.as_millis()).dimmed()
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────
//  sc export
// ──────────────────────────────────────────────────────────────

async fn cmd_export(
    client: &Client,
    host:   &str,
    out:    Option<PathBuf>,
) -> Result<(), String> {
    println!("{}", "◆ Exportando delta federado…".cyan().bold());

    let url = format!("{}/api/v1/federation/export", host);

    let resp = client
        .post(&url)
        .json(&json!({}))
        .send()
        .await
        .map_err(|e| format_reqwest_error(e, host))?;

    let status = resp.status();
    if !status.is_success() {
        let body: Value = resp.json().await.unwrap_or_default();
        let msg = body["error"].as_str().unwrap_or("error desconocido");
        return Err(format!("Error {}: {}", status.as_u16(), msg));
    }

    // Obtener el cuerpo como texto (JSON del delta).
    let body_text = resp.text().await.map_err(|e| e.to_string())?;

    match out {
        Some(path) => {
            tokio::fs::write(&path, &body_text)
                .await
                .map_err(|e| format!("No se pudo escribir '{}': {}", path.display(), e))?;
            println!(
                "{} Delta guardado en {}",
                "✓".green().bold(),
                path.display().to_string().yellow()
            );
        }
        None => {
            // Sin archivo de salida: volcar JSON pretty-printed a stdout.
            let pretty: Value = serde_json::from_str(&body_text)
                .unwrap_or_else(|_| Value::String(body_text));
            println!("{}", serde_json::to_string_pretty(&pretty).unwrap_or_default().white());
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────
//  sc watch
// ──────────────────────────────────────────────────────────────

async fn cmd_watch(host: &str) -> Result<(), String> {
    // Construir URL WebSocket: http → ws, https → wss.
    let ws_url = host
        .replacen("https://", "wss://", 1)
        .replacen("http://",  "ws://",  1);
    let ws_url = format!("{}/ws/subconscious", ws_url);

    println!(
        "{} {}",
        "◆ Conectando a telemetría en vivo:".cyan().bold(),
        ws_url.dimmed()
    );
    println!(
        "{}",
        "  Ctrl+C para desconectar.\n".dimmed()
    );

    // Abrir conexión WebSocket (rustls vía tokio-tungstenite).
    let (ws_stream, _) = connect_async(&ws_url)
        .await
        .map_err(|e| format_ws_error(e, host))?;

    println!("{}", "✓ Conectado — escuchando eventos…\n".green());

    let (_, mut read) = ws_stream.split();

    // Bucle de lectura: cada mensaje del servidor es un SubconsciousEvent JSON.
    loop {
        match read.next().await {
            Some(Ok(Message::Text(raw))) => {
                print_event(&raw);
            }
            Some(Ok(Message::Ping(_))) => {
                // Pings de keepalive — ignorar silenciosamente.
            }
            Some(Ok(Message::Close(frame))) => {
                let reason = frame
                    .as_ref()
                    .map(|f| f.reason.as_ref())
                    .unwrap_or("sin motivo");
                println!(
                    "\n{} El servidor cerró la conexión: {}",
                    "⚠".yellow(),
                    reason.dimmed()
                );
                break;
            }
            Some(Ok(_)) => {
                // Otros tipos de frame (Binary, etc.) — ignorar.
            }
            Some(Err(e)) => {
                return Err(format!("Error en WebSocket: {}", e));
            }
            None => {
                // Stream terminó (EOF).
                println!("\n{}", "  Conexión cerrada.".dimmed());
                break;
            }
        }
    }

    Ok(())
}

/// Formatea e imprime un evento del subconsciente con colores por tipo.
fn print_event(raw: &str) {
    // Intentar deserializar como SubconsciousEvent.
    if let Ok(event) = serde_json::from_str::<SubconsciousEvent>(raw) {
        // Timestamp: convertir epoch Unix a HH:MM:SS local aproximado.
        let ts = format_timestamp(event.timestamp);

        // Prefijo e icono según tipo de evento.
        let (icon, label, colorized_msg) = match event.event_type.as_str() {
            "Consolidation" => (
                "✓",
                "CONSOLIDACIÓN".green().bold(),
                event.message.green().to_string(),
            ),
            "Validation" => (
                "◉",
                "VALIDACIÓN   ".cyan(),
                event.message.cyan().to_string(),
            ),
            "Hypothesis" => (
                "?",
                "HIPÓTESIS    ".blue().bold(),
                event.message.blue().to_string(),
            ),
            "Rejection" => (
                "✗",
                "RECHAZO      ".yellow(),
                event.message.yellow().to_string(),
            ),
            "Error" => (
                "⚠",
                "ERROR        ".red().bold(),
                event.message.red().to_string(),
            ),
            "Lag" => (
                "~",
                "LAG          ".magenta(),
                event.message.magenta().to_string(),
            ),
            _ => (
                "·",
                "CICLO        ".dimmed(),
                event.message.dimmed().to_string(),
            ),
        };

        println!(
            "{} {} {} {}",
            ts.dimmed(),
            icon,
            label,
            colorized_msg
        );
    } else {
        // Mensaje no parseable: mostrar raw en gris.
        println!("{} {}", "·".dimmed(), raw.dimmed());
    }
}

/// Convierte un epoch Unix (segundos) a una cadena HH:MM:SS.
/// Implementación sin deps extra: usa aritmética simple.
fn format_timestamp(epoch: u64) -> String {
    let secs  = epoch % 86_400; // segundos dentro del día actual
    let h     = secs / 3_600;
    let m     = (secs % 3_600) / 60;
    let s     = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

// ──────────────────────────────────────────────────────────────
//  sc health
// ──────────────────────────────────────────────────────────────

async fn cmd_health(client: &Client, host: &str) -> Result<(), String> {
    let url = format!("{}/health", host);

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format_reqwest_error(e, host))?;

    let status_code = resp.status();
    if !status_code.is_success() {
        return Err(format!("Error HTTP {}", status_code.as_u16()));
    }

    let data: HealthResponse = resp.json().await.map_err(|e| e.to_string())?;

    println!("{}", "┌─── SynaptoCore Health ───────────────────┐".cyan());
    println!(
        "{}  {:<12} {}",
        "│".cyan(),
        "Estado:".dimmed(),
        data.status.green().bold()
    );
    println!(
        "{}  {:<12} {}",
        "│".cyan(),
        "Nodos:".dimmed(),
        data.total_nodes.to_string().yellow()
    );
    println!(
        "{}  {:<12} {}",
        "│".cyan(),
        "Aristas:".dimmed(),
        data.total_edges.to_string().yellow()
    );
    println!(
        "{}  {:<12} {}",
        "│".cyan(),
        "Host:".dimmed(),
        host.white()
    );
    println!("{}", "└──────────────────────────────────────────┘".cyan());

    Ok(())
}

// ──────────────────────────────────────────────────────────────
//  Helpers de formateo de errores (UX limpio)
// ──────────────────────────────────────────────────────────────

/// Convierte un error de `reqwest` en un mensaje de usuario legible.
/// Si es un error de conexión (Connection Refused), da instrucciones claras.
fn format_reqwest_error(e: reqwest::Error, host: &str) -> String {
    if e.is_connect() || e.is_request() {
        // El servidor no está en línea — mensaje amigable.
        format!(
            "Error: El cerebro SynaptoCore no está en línea en {}\n\
             → Inicia el servidor con: cargo run --release",
            host
        )
    } else if e.is_timeout() {
        format!(
            "Error: Timeout esperando respuesta de {} (>30s)\n\
             → El servidor puede estar sobrecargado.",
            host
        )
    } else {
        format!("Error HTTP: {}", e)
    }
}

/// Convierte un error de WebSocket en un mensaje de usuario legible.
fn format_ws_error(e: tokio_tungstenite::tungstenite::Error, host: &str) -> String {
    let msg = e.to_string();
    if msg.contains("Connection refused") || msg.contains("Os error") {
        format!(
            "Error: No se pudo conectar al WebSocket en {}\n\
             → El cerebro SynaptoCore no está en línea.\n\
             → Inicia el servidor con: cargo run --release",
            host
        )
    } else {
        format!("Error WebSocket: {}", msg)
    }
}
