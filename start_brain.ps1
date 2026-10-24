#!/usr/bin/env pwsh
# =============================================================================
# start_brain.ps1 — Script de Arranque Rápido de SynaptoCore
# =============================================================================
# Uso:
#   .\start_brain.ps1                      # uso normal: Ollama + búsqueda semántica
#
# Opciones experimentales (compilan los enlaces con llama.cpp; el servidor
# sigue usando Ollama como motor):
#   .\start_brain.ps1 -Gpu                 # llama.cpp + CUDA (NVIDIA)
#   .\start_brain.ps1 -Vulkan              # llama.cpp + Vulkan (AMD / Intel)
#
# -Embeddings se mantiene por compatibilidad: la búsqueda semántica ya viene
# activada por defecto.
# =============================================================================

param(
    [switch]$Gpu,       # Experimental: compila llama.cpp con CUDA (NVIDIA)
    [switch]$Vulkan,    # Experimental: compila llama.cpp con Vulkan (AMD / Intel Arc)
    [switch]$Embeddings # Compatibilidad: la búsqueda semántica ya es la opción por defecto
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          SynaptoCore v0.17 — Autonomous Cognitive Brain          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Cargar variables de entorno desde .env ─────────────────────────────────
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Write-Host "✓ Cargando variables de entorno desde .env..." -ForegroundColor Green
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        # Ignorar comentarios y líneas vacías
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Count -eq 2) {
                $key   = $parts[0].Trim()
                $value = $parts[1].Trim()
                [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
                Write-Host "  → $key = $value" -ForegroundColor DarkGray
            }
        }
    }
    Write-Host ""
} else {
    Write-Host "⚠ No se encontró .env — usando valores por defecto." -ForegroundColor Yellow
    Write-Host "  Consejo: copia .env.example a .env y ajusta tus valores." -ForegroundColor DarkGray
    Write-Host ""
}

# ── 2. Construir la lista de features ─────────────────────────────────────────
$features = @()

if ($Gpu -and $Vulkan) {
    Write-Host "⚠ -Gpu y -Vulkan son mutuamente excluyentes. Usando -Gpu (CUDA)." -ForegroundColor Yellow
    $features += "cuda"
    $features += "llama-backend"
} elseif ($Gpu) {
    Write-Host "🚀 Modo GPU NVIDIA CUDA activado." -ForegroundColor Magenta
    $features += "cuda"
    $features += "llama-backend"
} elseif ($Vulkan) {
    Write-Host "🚀 Modo GPU Vulkan activado." -ForegroundColor Magenta
    $features += "vulkan"
    $features += "llama-backend"
} else {
    Write-Host "💡 Motor LLM: Ollama (usa la GPU automáticamente si está disponible)." -ForegroundColor Yellow
}

if ($Embeddings) {
    Write-Host "🧠 Búsqueda semántica vectorial (fastembed) activada." -ForegroundColor Cyan
    $features += "embeddings"
}

# ── 3. Construir el comando cargo run ─────────────────────────────────────────
$cargoArgs = @("run", "--release", "--bin", "synaptocore")

if ($features.Count -gt 0) {
    $featureStr = $features -join ","
    $cargoArgs += "--features"
    $cargoArgs += $featureStr
}

# ── 4. Mostrar configuración final ────────────────────────────────────────────
Write-Host ""
Write-Host "Configuración:" -ForegroundColor White
$port = if ($env:SYNAPTOCORE_PORT) { $env:SYNAPTOCORE_PORT } else { "8080" }
Write-Host "  Panel/API : http://127.0.0.1:$port/" -ForegroundColor DarkGray
Write-Host "  Modelo    : $(if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { 'qwen3.5:2b-q4_K_M' }) (Ollama)" -ForegroundColor DarkGray
if ($env:SYNAPTOCORE_MODEL_PATH) {
    Write-Host "  Modelo LLM: $($env:SYNAPTOCORE_MODEL_PATH)" -ForegroundColor DarkGray
}
if ($env:SYNAPTOCORE_CONNECTORS) {
    Write-Host "  Conectores: $($env:SYNAPTOCORE_CONNECTORS)" -ForegroundColor DarkGray
}
if ($env:SEARXNG_URL) {
    Write-Host "  SearXNG   : $($env:SEARXNG_URL)" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "Ejecutando: cargo $($cargoArgs -join ' ')" -ForegroundColor Green
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# ── 5. Ejecutar SynaptoCore ───────────────────────────────────────────────────
Push-Location $PSScriptRoot
try {
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ SynaptoCore terminó con código de error: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
