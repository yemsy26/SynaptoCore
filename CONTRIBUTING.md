# Cómo contribuir

Gracias por tu interés en SynaptoCore. Esta guía explica cómo preparar el entorno y cómo proponer cambios.

## Antes de empezar

- Para errores y propuestas, abre un [issue](https://github.com/yemsy26/SynaptoCore/issues) con las plantillas disponibles.
- Para cambios grandes (un módulo nuevo, un cambio de arquitectura o de la API), abre primero un issue para acordar el enfoque.
- Las vulnerabilidades no se reportan en issues públicos: consulta [SECURITY.md](SECURITY.md).

## Entorno de desarrollo

1. Instala Rust estable con [rustup](https://rustup.rs).
2. Clona tu fork y crea una rama a partir de `main`:

   ```bash
   git clone https://github.com/<tu-usuario>/SynaptoCore.git
   cd SynaptoCore
   git checkout -b mi-cambio
   ```

3. Comprueba que todo compila y pasa las pruebas:

   ```bash
   cargo test --no-default-features   # rápido, sin descargas
   cargo test                         # suite completa, con embeddings
   ```

Las pruebas no necesitan Ollama, SearXNG ni conexión: usan modelos simulados y bases de datos en memoria. Para probar a mano con servicios reales, sigue la [instalación del README](README.md#instalación) y usa una copia de tu base de datos, nunca la original.

## Convenciones

- **Idioma:** los comentarios, los mensajes al usuario y la documentación se escriben en español.
- **Estilo:** sigue el estilo del archivo que modificas. Ejecuta `cargo fmt` sobre el código nuevo y revisa `cargo clippy`.
- **Pruebas:** todo cambio de comportamiento lleva su prueba. Las que dependen de servicios externos se marcan con `#[ignore]` y explican cómo ejecutarlas.
- **Concurrencia:** las llamadas al modelo, la red y el cálculo de embeddings se hacen sin tener bloqueado el grafo; el bloqueo se toma solo para leer o escribir en él.
- **Procedencia:** todo conocimiento nuevo se guarda con su origen (`EdgeOrigin`). Nunca se aprende de lo que redacta el propio modelo, solo de fuentes o de lo que enseña el usuario.
- **Esquema:** los cambios en la base de datos deben ser compatibles con bases existentes (migraciones idempotentes en `run_migrations`).

## Commits y pull requests

- Haz commits pequeños y con un mensaje que explique el porqué, no solo el qué.
- Antes de abrir el pull request, ejecuta las pruebas en las dos configuraciones (`cargo test` y `cargo test --no-default-features`).
- Describe en el pull request qué cambia, cómo lo probaste y si afecta a la API o al esquema de la base de datos.
- Si el cambio afecta al uso, actualiza el [README](README.md) o la [referencia de la API](docs/API.md).

## Licencia

Al contribuir aceptas que tu aportación se distribuya bajo la [licencia MIT](LICENSE) del proyecto.
