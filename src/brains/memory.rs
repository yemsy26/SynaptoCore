// brains/memory.rs — Paso 33b: ¿caben los cerebros en memoria?
//
// Con dos modelos grandes cargados a la vez (un 9B y un 35B en Bionic) un
// equipo de 32 GB se quedó con 0,2 GB libres y Ollama ya no pudo arrancar.
// Aquí se mide la RAM y la VRAM libres y se estima cuánto más hará falta al
// cargar los cerebros elegidos, para avisar antes de elegirlos.
//
// Lo que un modelo ya cargado ocupa está dentro de lo usado: solo cuentan
// los que aún hay que cargar. Un modelo se puede repartir entre la GPU y la
// RAM, así que lo disponible es la suma de las dos.

use std::time::Duration;

use serde::Serialize;

use super::{BrainInfo, BrainsConfig};

/// Memoria de trabajo además de los pesos (caché KV, buffers del motor).
const OVERHEAD_FACTOR: f64 = 1.15;
const OVERHEAD_BYTES: u64 = 512 * MIB;
/// Por encima de esta fracción de lo disponible, la carga va justa.
const TIGHT_FRACTION: f64 = 0.8;
/// Con menos RAM libre que esto, el equipo ya va justo aunque no haya que
/// cargar nada.
const LOW_RAM_BYTES: u64 = 1536 * MIB;
const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_secs(3);

const MIB: u64 = 1024 * 1024;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// RAM y VRAM del equipo, en bytes. `None` si no se pudo medir.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryInfo {
    pub ram_total:  Option<u64>,
    pub ram_free:   Option<u64>,
    pub vram_total: Option<u64>,
    pub vram_free:  Option<u64>,
    pub gpu:        Option<String>,
}

/// Mide la memoria ahora mismo.
pub fn memory_info() -> MemoryInfo {
    let (ram_total, ram_free) = ram().map_or((None, None), |(t, f)| (Some(t), Some(f)));
    let gpu = nvidia_gpu();
    MemoryInfo {
        ram_total,
        ram_free,
        vram_total: gpu.as_ref().map(|g| g.1),
        vram_free:  gpu.as_ref().map(|g| g.2),
        gpu:        gpu.map(|g| g.0),
    }
}

#[cfg(windows)]
fn ram() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemoryStatusEx {
        length:                 u32,
        memory_load:            u32,
        total_phys:             u64,
        avail_phys:             u64,
        total_page_file:        u64,
        avail_page_file:        u64,
        total_virtual:          u64,
        avail_virtual:          u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0, total_phys: 0, avail_phys: 0, total_page_file: 0, avail_page_file: 0,
        total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0,
    };
    // SAFETY: `status` es un MEMORYSTATUSEX válido con `dwLength` puesto,
    // como pide la API; la llamada solo escribe en él.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    ok.then_some((status.total_phys, status.avail_phys))
}

#[cfg(target_os = "linux")]
fn ram() -> Option<(u64, u64)> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn ram() -> Option<(u64, u64)> {
    None
}

/// `MemTotal` y `MemAvailable` de `/proc/meminfo`, en bytes.
#[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(name))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    };
    Some((field("MemTotal:")?, field("MemAvailable:")?))
}

/// Nombre, VRAM total y libre de la primera GPU NVIDIA (`nvidia-smi`).
fn nvidia_gpu() -> Option<(String, u64, u64)> {
    let mut child = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut stdout, &mut text);
        let _ = tx.send(text);
    });
    let text = rx.recv_timeout(NVIDIA_SMI_TIMEOUT).ok();
    let _ = child.kill();
    let _ = child.wait();
    parse_nvidia_smi(&text?)
}

/// «NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 312» (MiB) → bytes.
fn parse_nvidia_smi(text: &str) -> Option<(String, u64, u64)> {
    let mut fields = text.lines().next()?.split(',').map(str::trim);
    let name = fields.next()?.to_string();
    let total: u64 = fields.next()?.parse().ok()?;
    let free: u64 = fields.next()?.parse().ok()?;
    Some((name, total * MIB, free * MIB))
}

/// Cuánto ocupa un modelo al cargarlo: su archivo o, sin él, lo que da su
/// tamaño y compresión.
pub fn model_bytes(info: &BrainInfo) -> Option<u64> {
    info.size_bytes.or_else(|| {
        let params = info.params_b? as f64 * 1e9;
        Some((params * info.bits.unwrap_or(4.0) as f64 / 8.0) as u64)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitStatus {
    /// Caben con margen.
    Ok,
    /// Caben, pero dejan el equipo justo de memoria.
    Justo,
    /// No caben: el equipo se quedará sin memoria.
    NoCabe,
    /// No se pudo medir la memoria.
    Desconocido,
}

/// ¿Caben los cerebros elegidos?
#[derive(Debug, Clone, Serialize)]
pub struct MemoryFit {
    pub status:          FitStatus,
    /// Memoria que harán falta los cerebros que aún no están cargados.
    pub need_bytes:      u64,
    /// RAM libre más VRAM libre.
    pub available_bytes: Option<u64>,
    pub message:         String,
    /// Modelos cargados que no son cerebros: ocupan memoria («qwythos 9B · 6,7 GB»).
    pub idle_loaded:     Vec<String>,
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / GIB).replace('.', ",")
}

/// Estima si los cerebros de `config` caben con la memoria de `mem`.
pub fn memory_fit(config: &BrainsConfig, catalog: &[BrainInfo], mem: &MemoryInfo) -> MemoryFit {
    let is_chosen = |b: &BrainInfo| [&config.rapido, &config.profundo]
        .iter()
        .any(|c| c.engine == b.engine && c.model.eq_ignore_ascii_case(&b.model));
    // El mismo modelo en las dos funciones se carga una vez.
    let to_load: Vec<(&BrainInfo, u64)> = catalog.iter()
        .filter(|b| is_chosen(b) && !b.loaded)
        .filter_map(|b| model_bytes(b).map(|bytes| (b, (bytes as f64 * OVERHEAD_FACTOR) as u64 + OVERHEAD_BYTES)))
        .collect();
    let need_bytes: u64 = to_load.iter().map(|(_, bytes)| bytes).sum();
    let largest = to_load.iter().max_by_key(|(_, bytes)| *bytes)
        .map(|(b, bytes)| format!("«{}» ({})", b.model, gb(*bytes)))
        .unwrap_or_default();
    let idle_loaded: Vec<String> = catalog.iter()
        .filter(|b| b.loaded && !is_chosen(b))
        .map(|b| match model_bytes(b) {
            Some(bytes) => format!("{} ({}) · {}", b.model, b.engine_label, gb(bytes)),
            None => format!("{} ({})", b.model, b.engine_label),
        })
        .collect();
    let idle_note = if idle_loaded.is_empty() {
        String::new()
    } else {
        " Hay modelos cargados que no son cerebros: descárgalos para liberar memoria.".to_string()
    };

    let Some(ram_free) = mem.ram_free else {
        return MemoryFit {
            status: FitStatus::Desconocido,
            need_bytes,
            available_bytes: None,
            message: "No se pudo medir la memoria del equipo.".to_string(),
            idle_loaded,
        };
    };
    let available = ram_free + mem.vram_free.unwrap_or(0);
    let (status, message) = if need_bytes > available {
        (FitStatus::NoCabe, format!(
            "No caben: cargarlos necesita unos {} más y solo quedan {} libres (RAM + VRAM). \
             El equipo se quedará sin memoria y Ollama o Bionic pueden dejar de responder. \
             El que más ocupa es {largest}: elige uno más pequeño, usa el mismo modelo para los dos \
             o cierra programas para liberar memoria.{idle_note}",
            gb(need_bytes), gb(available)))
    } else if need_bytes as f64 > available as f64 * TIGHT_FRACTION {
        (FitStatus::Justo, format!(
            "Caben justos: necesitan unos {} de los {} libres. El equipo irá lento mientras estén cargados.{idle_note}",
            gb(need_bytes), gb(available)))
    } else if need_bytes == 0 && ram_free < LOW_RAM_BYTES {
        (FitStatus::Justo, format!(
            "Ya están cargados, pero solo quedan {} de RAM libre.{idle_note}", gb(ram_free)))
    } else if need_bytes == 0 {
        (FitStatus::Ok, format!("Ya están cargados; quedan {} libres.{idle_note}", gb(available)))
    } else {
        (FitStatus::Ok, format!(
            "Caben: necesitan unos {} de los {} libres.{idle_note}", gb(need_bytes), gb(available)))
    };
    MemoryFit { status, need_bytes, available_bytes: Some(available), message, idle_loaded }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brains::{BrainChoice, Tier};
    use crate::openai_backend::Reasoning;

    const GB: u64 = 1024 * 1024 * 1024;

    fn brain(engine: &str, model: &str, size_gb: f64, loaded: bool) -> BrainInfo {
        BrainInfo {
            engine: engine.into(), engine_label: engine.into(), model: model.into(), display: model.into(),
            params: None, params_b: None, quantization: None, bits: None, context: None,
            size_bytes: Some((size_gb * GB as f64) as u64), loaded, reasoning: Reasoning::Unknown, tier: Tier::Ligero,
        }
    }

    fn config(rapido: (&str, &str), profundo: (&str, &str)) -> BrainsConfig {
        let choice = |(engine, model): (&str, &str)| BrainChoice { engine: engine.into(), model: model.into(), tier: None };
        BrainsConfig { rapido: choice(rapido), profundo: choice(profundo) }
    }

    fn memory(ram_free_gb: u64, vram_free_gb: u64) -> MemoryInfo {
        MemoryInfo { ram_total: Some(32 * GB), ram_free: Some(ram_free_gb * GB), vram_total: Some(8 * GB),
                     vram_free: Some(vram_free_gb * GB), gpu: Some("RTX".into()) }
    }

    fn catalog() -> Vec<BrainInfo> {
        vec![brain("ollama", "qwen2b", 1.9, false), brain("bionic", "qwythos9b", 6.7, false),
             brain("bionic", "qwen35b", 17.7, false), brain("ollama", "llama8b", 4.9, true)]
    }

    #[test]
    fn test_two_big_brains_do_not_fit() {
        // Lo que pasó: 9B y 35B a la vez con ~20 GB libres.
        let fit = memory_fit(&config(("bionic", "qwythos9b"), ("bionic", "qwen35b")), &catalog(), &memory(18, 2));
        assert_eq!(fit.status, FitStatus::NoCabe, "{}", fit.message);
        assert!(fit.message.contains("No caben") && fit.message.contains("«qwen35b» (20,9 GB)"), "{}", fit.message);
        assert_eq!(fit.idle_loaded, ["llama8b (ollama) · 4,9 GB"], "el 8B cargado ocupa sin ser cerebro");
    }

    #[test]
    fn test_small_fast_brain_with_big_deep_brain() {
        let fit = memory_fit(&config(("ollama", "qwen2b"), ("bionic", "qwen35b")), &catalog(), &memory(18, 7));
        assert_eq!(fit.status, FitStatus::Justo, "{}", fit.message);
        let fit = memory_fit(&config(("ollama", "qwen2b"), ("bionic", "qwen35b")), &catalog(), &memory(28, 7));
        assert_eq!(fit.status, FitStatus::Ok, "{}", fit.message);
    }

    #[test]
    fn test_same_model_counts_once_and_loaded_needs_nothing() {
        let once = memory_fit(&config(("ollama", "qwen2b"), ("ollama", "qwen2b")), &catalog(), &memory(20, 7));
        let size = (1.9 * GB as f64) as u64;
        let single = (size as f64 * OVERHEAD_FACTOR) as u64 + OVERHEAD_BYTES;
        assert_eq!(once.need_bytes, single);
        let loaded = memory_fit(&config(("ollama", "llama8b"), ("ollama", "llama8b")), &catalog(), &memory(20, 7));
        assert_eq!((loaded.need_bytes, loaded.status), (0, FitStatus::Ok));
        assert!(loaded.idle_loaded.is_empty());
        let low = memory_fit(&config(("ollama", "llama8b"), ("ollama", "llama8b")), &catalog(), &memory(1, 0));
        assert_eq!(low.status, FitStatus::Justo, "sin nada que cargar, pero con la RAM casi agotada");
    }

    #[test]
    fn test_unknown_memory() {
        let fit = memory_fit(&config(("ollama", "qwen2b"), ("ollama", "qwen2b")), &catalog(), &MemoryInfo::default());
        assert_eq!(fit.status, FitStatus::Desconocido);
    }

    #[test]
    fn test_model_bytes_without_file_size() {
        let mut b = brain("x", "m", 0.0, false);
        b.size_bytes = None;
        b.params_b = Some(8.0);
        b.bits = Some(4.0);
        assert_eq!(model_bytes(&b), Some(4_000_000_000));
        b.params_b = None;
        assert_eq!(model_bytes(&b), None);
    }

    #[test]
    fn test_parsers() {
        assert_eq!(parse_meminfo("MemTotal:       32000000 kB\nMemFree: 1 kB\nMemAvailable:   1000 kB\n"),
                   Some((32_000_000 * 1024, 1000 * 1024)));
        assert_eq!(parse_meminfo("MemTotal: 5 kB\n"), None);
        assert_eq!(parse_nvidia_smi("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 312\n"),
                   Some(("NVIDIA GeForce RTX 4060 Laptop GPU".to_string(), 8188 * MIB, 312 * MIB)));
        assert_eq!(parse_nvidia_smi("No devices were found"), None);
    }

    #[test]
    fn test_memory_info_does_not_fail() {
        let info = memory_info();
        if cfg!(any(windows, target_os = "linux")) {
            assert!(info.ram_total.is_some() && info.ram_free.is_some(), "{info:?}");
        }
    }
}
