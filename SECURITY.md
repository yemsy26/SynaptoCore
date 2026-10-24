# Política de seguridad

## Versiones con soporte

Solo la última versión publicada en la rama `main` recibe correcciones de seguridad.

## Cómo reportar una vulnerabilidad

No abras un issue público. Usa el reporte privado de GitHub: pestaña **Security** del repositorio → **Report a vulnerability**.

Incluye, si es posible:

- una descripción del problema y su impacto;
- los pasos para reproducirlo;
- la versión o el commit afectado.

Recibirás una primera respuesta en un plazo de siete días.

## Recomendaciones de despliegue

- El servidor escucha solo en `127.0.0.1` y no tiene autenticación: no lo expongas a la red sin un proxy inverso con autenticación.
- Cambia `secret_key` en `searxng-settings.yml` antes de exponer SearXNG fuera de tu equipo.
- Guarda los tokens de los conectores en variables de entorno (`${NOMBRE}` en `conectores.json`), nunca en el archivo, y no subas `.env` ni `conectores.json` al repositorio.
- `synaptocore_memory.db` contiene todo lo aprendido: trátalo como un dato privado.
