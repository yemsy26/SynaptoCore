@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: start_brain.bat — SynaptoCore v0.17 Quick-Start (Windows)
:: ============================================================================
:: Flujo:
::   [1] Validar dependencias (Rust, Ollama)
::   [2] Docker: levantar SearXNG si Docker esta disponible
::   [3] Lanzar servidor Rust en ventana separada (async)
::   [4] Polling HTTP hasta que la API responda
::   [5] Abrir navegador y cerrar este iniciador
:: ============================================================================

:: Cambiar al directorio del .bat (critico para doble-clic)
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"

title SynaptoCore v0.17 - Iniciador

echo.
echo  +=================================================================+
echo  ^|        SynaptoCore v0.17 - Autonomous Cognitive Brain           ^|
echo  +=================================================================+
echo.

:: ── Inyectar PATH de Rust/Cargo ─────────────────────────────────────────
set "CARGO_HOME_BIN=%USERPROFILE%\.cargo\bin"
if exist "%CARGO_HOME_BIN%\cargo.exe" set "PATH=%CARGO_HOME_BIN%;%PATH%"

:: ── Inyectar PATH de Ollama ──────────────────────────────────────────────
set "OLLAMA_BIN=%LOCALAPPDATA%\Programs\Ollama"
if exist "%OLLAMA_BIN%\ollama.exe" set "PATH=%OLLAMA_BIN%;%PATH%"

:: ── Leer flags de linea de comandos ──────────────────────────────────────
set "USE_GPU=0"
set "USE_VULKAN=0"

for %%A in (%*) do (
    if /I "%%A"=="gpu"    set "USE_GPU=1"
    if /I "%%A"=="vulkan" set "USE_VULKAN=1"
)

:: ══════════════════════════════════════════════════════════════════════════
:: [1/5] CONFIGURACION DE ENTORNO
:: ══════════════════════════════════════════════════════════════════════════
echo  [1/5] Configuracion de entorno...

if exist "%SCRIPT_DIR%.env" (
    echo        Cargando .env...
    for /f "usebackq eol=# tokens=1,* delims==" %%K in ("%SCRIPT_DIR%.env") do (
        if not "%%K"=="" set "%%K=%%L"
    )
    echo        .env cargado OK
) else (
    echo        .env no encontrado - usando valores por defecto.
    echo        TIP: copia .env.example a .env y ajusta los valores.
)

:: Valores por defecto
if not defined OLLAMA_HOST  set "OLLAMA_HOST=http://localhost:11434"
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=qwen3.5:2b-q4_K_M"
if not defined SC_HOST      set "SC_HOST=127.0.0.1"
if not defined SC_PORT      set "SC_PORT=8080"
if not defined SC_DB_PATH   set "SC_DB_PATH=synaptocore_memory.db"
echo.

:: ══════════════════════════════════════════════════════════════════════════
:: [2/5] VALIDAR DEPENDENCIAS
:: ══════════════════════════════════════════════════════════════════════════
echo  [2/5] Verificando dependencias...
echo.

:: -- Rust / Cargo --
where cargo >nul 2>&1
if errorlevel 1 (
    echo  +------------------------------------------------------------+
    echo  ^|  ERROR: Rust/Cargo no encontrado en PATH.                  ^|
    echo  ^|                                                             ^|
    echo  ^|  1. Instala Rust: https://rustup.rs/                       ^|
    echo  ^|  2. Reinicia esta ventana y ejecuta start_brain.bat        ^|
    echo  +------------------------------------------------------------+
    echo.
    goto FATAL_ERROR
)
for /f "tokens=1-3" %%A in ('cargo --version 2^>nul') do set "CARGO_VER=%%A %%B %%C"
echo        Rust/Cargo   : OK  (!CARGO_VER!)

if exist "%SCRIPT_DIR%target\release\synaptocore.exe" (
    echo        Binario      : ya compilado
) else (
    echo        Binario      : pendiente  ^(primer build ~2-5 min^)
)

:: -- Ollama --
where ollama >nul 2>&1
set "OLLAMA_OK=%errorlevel%"
if "%OLLAMA_OK%"=="0" (
    echo        Ollama       : OK
    echo        Modelo LLM   : %OLLAMA_MODEL%
) else (
    echo        Ollama       : no instalado  ^(se usara ServerMock^)
    echo                       https://ollama.com
)
echo.

:: ══════════════════════════════════════════════════════════════════════════
:: [3/5] DOCKER / SearXNG
:: ══════════════════════════════════════════════════════════════════════════
echo  [3/5] Verificando SearXNG ^(motor de busqueda epistemico^)...

where docker >nul 2>&1
if errorlevel 1 (
    echo        Docker       : no instalado - SearXNG desactivado.
    echo                       El subconsciente operara sin busqueda web.
    echo                       Instala Docker: https://www.docker.com
    if not defined SEARXNG_URL set "SEARXNG_URL="
) else (
    :: Docker esta en PATH pero puede que el daemon no este corriendo.
    docker info >nul 2>&1
    if errorlevel 1 (
        echo        Docker       : instalado pero daemon no responde.
        echo                       Abre Docker Desktop y vuelve a intentarlo.
        echo                       SearXNG desactivado por ahora.
        if not defined SEARXNG_URL set "SEARXNG_URL="
    ) else (
        echo        Docker       : OK

        :: Verificar estado del contenedor searxng-brain
        set "CONTAINER_STATUS="
        for /f "delims=" %%S in ('docker inspect --format "{{.State.Status}}" searxng-brain 2^>nul') do (
            set "CONTAINER_STATUS=%%S"
        )

        if "!CONTAINER_STATUS!"=="running" (
            echo        SearXNG      : ya corriendo  ^(contenedor: searxng-brain^)
        ) else if "!CONTAINER_STATUS!"=="exited" (
            echo        SearXNG      : contenedor detenido - reiniciando...
            docker start searxng-brain >nul 2>&1
            if errorlevel 1 (
                echo        SearXNG      : ERROR al reiniciar contenedor.
            ) else (
                echo        SearXNG      : reiniciado OK
            )
        ) else (
            echo        SearXNG      : creando contenedor nuevo...
            set "DOCKER_MOUNTS="
            if exist "%SCRIPT_DIR%searxng-settings.yml" (
                set "DOCKER_MOUNTS=-v ""%SCRIPT_DIR%searxng-settings.yml:/etc/searxng/settings.yml"""
            )
            if exist "%SCRIPT_DIR%searxng-limiter.toml" (
                set "DOCKER_MOUNTS=!DOCKER_MOUNTS! -v ""%SCRIPT_DIR%searxng-limiter.toml:/etc/searxng/limiter.toml"""
            )
            docker run -d --name searxng-brain -p 8888:8080 --restart unless-stopped !DOCKER_MOUNTS! searxng/searxng >nul 2>&1
            if errorlevel 1 (
                echo        SearXNG      : ERROR al crear contenedor.
                echo                       Verifica que el puerto 8888 este libre.
            ) else (
                echo        SearXNG      : contenedor creado y levantado OK
            )
        )

        :: Asignar URL si no viene del .env
        if not defined SEARXNG_URL set "SEARXNG_URL=http://localhost:8888"
        echo        SEARXNG_URL  : %SEARXNG_URL%
    )
)
echo.

:: ══════════════════════════════════════════════════════════════════════════
:: [4/5] CONSTRUIR COMANDO CARGO Y LANZAR SERVIDOR (async)
:: ══════════════════════════════════════════════════════════════════════════
set "FEAT_LIST="
set "HAS_FEAT=0"

if "%USE_GPU%"=="1" (
    if "%USE_VULKAN%"=="1" (
        echo  AVISO: gpu y vulkan son excluyentes. Se usa gpu ^(CUDA^).
        set "USE_VULKAN=0"
    )
    set "FEAT_LIST=cuda,llama-backend"
    set "HAS_FEAT=1"
) else if "%USE_VULKAN%"=="1" (
    set "FEAT_LIST=vulkan,llama-backend"
    set "HAS_FEAT=1"
)

set "CARGO_CMD=cargo run --release --bin synaptocore"
if "!HAS_FEAT!"=="1" set "CARGO_CMD=!CARGO_CMD! --features !FEAT_LIST!"

echo  [4/5] Lanzando servidor SynaptoCore...
echo        Comando : !CARGO_CMD!
echo.

:: Exportar variables al entorno para que el proceso hijo las herede.
:: SET en CMD siempre es local a esta sesion, que hereda el proceso lanzado.
set "OLLAMA_HOST=%OLLAMA_HOST%"
set "OLLAMA_MODEL=%OLLAMA_MODEL%"
set "SEARXNG_URL=%SEARXNG_URL%"
set "SC_HOST=%SC_HOST%"
set "SC_PORT=%SC_PORT%"
set "SC_DB_PATH=%SC_DB_PATH%"

:: Lanzar el servidor en una nueva ventana que se cierra sola con /c.
:: /c ejecuta el comando y cierra al terminar (o al Ctrl+C).
:: La ventana quedara abierta mostrando logs mientras el servidor corra.
start "SynaptoCore - Servidor" cmd /k "cd /d %SCRIPT_DIR% && echo Iniciando SynaptoCore... && !CARGO_CMD!"

:: ══════════════════════════════════════════════════════════════════════════
:: [5/5] POLLING HTTP + ABRIR NAVEGADOR
:: ══════════════════════════════════════════════════════════════════════════
echo  [5/5] Esperando que el servidor responda en http://%SC_HOST%:%SC_PORT%/health
echo        ^(Si es el primer build, espera 2-5 minutos^)
echo.

set "HEALTH_URL=http://%SC_HOST%:%SC_PORT%/health"
set "MAX_INTENTOS=90"
set "INTENTO=0"
set "SERVIDOR_LISTO=0"

:POLL_LOOP
set /a INTENTO+=1
if !INTENTO! GTR %MAX_INTENTOS% goto POLL_TIMEOUT

:: Polling via PowerShell — mas fiable que curl en Windows sin curl nativo.
powershell -NoProfile -NonInteractive -Command ^
  "try { $r = Invoke-WebRequest -Uri '%HEALTH_URL%' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if not errorlevel 1 (
    set "SERVIDOR_LISTO=1"
    goto SERVIDOR_UP
)

:: Mostrar progreso cada 5 intentos para no saturar la pantalla.
set /a MOD5=!INTENTO! %% 5
if "!MOD5!"=="0" (
    echo        Intento !INTENTO!/%MAX_INTENTOS% - compilando o iniciando...
)

:: Esperar 2 segundos antes del siguiente intento.
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 2" >nul 2>&1
goto POLL_LOOP

:POLL_TIMEOUT
echo.
echo  El servidor no respondio en !MAX_INTENTOS! intentos ^(~3 min^).
echo  El servidor sigue compilando en la otra ventana.
echo.
echo  Cuando veas el banner de SynaptoCore alli, abre manualmente:
echo    http://%SC_HOST%:%SC_PORT%/
echo.
goto DONE

:SERVIDOR_UP
echo.
echo  +================================================================+
echo  ^|  SynaptoCore listo!                                            ^|
echo  ^|                                                                ^|
echo  ^|  API    : http://%SC_HOST%:%SC_PORT%/health                   ^|
echo  ^|  Query  : POST http://%SC_HOST%:%SC_PORT%/api/v1/query        ^|
echo  ^|  Learn  : POST http://%SC_HOST%:%SC_PORT%/api/v1/learn        ^|
if defined SEARXNG_URL (
    echo  ^|  SearXNG: %SEARXNG_URL%                                ^|
)
echo  +================================================================+
echo.

:: Abrir el navegador predeterminado.
start "" "http://%SC_HOST%:%SC_PORT%/"

echo  Navegador abierto. Esta ventana se cerrara en 5 segundos...
powershell -NoProfile -NonInteractive -Command "Start-Sleep -Seconds 5" >nul 2>&1
goto DONE

:: ══════════════════════════════════════════════════════════════════════════
:FATAL_ERROR
echo  Inicio abortado. Corrige el error y ejecuta start_brain.bat de nuevo.
echo.

:DONE
echo  Presiona cualquier tecla para cerrar este iniciador...
pause >nul
endlocal
