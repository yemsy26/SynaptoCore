# Política de seguridad

## Versiones con soporte

Las correcciones de seguridad se publican en la rama `main`. Usa siempre la versión más reciente.

## Reportar una vulnerabilidad

Si encuentras un problema de seguridad, repórtalo de forma privada desde la pestaña **Security** de este repositorio, con el botón **Report a vulnerability**. No lo publiques en un issue.

Para que podamos reproducirlo, incluye:

- una descripción del problema y de su impacto;
- los pasos para reproducirlo;
- la versión o el commit afectado.

Recibirás una primera respuesta en un plazo de siete días.

## Uso seguro

SynaptoCore está pensado para ejecutarse en tu propio equipo:

- **Acceso local.** El servidor escucha solo en `127.0.0.1` y no incluye autenticación. Si necesitas usarlo desde otra máquina, colócalo detrás de un proxy inverso con autenticación.
- **SearXNG.** La `secret_key` de `searxng-settings.yml` es un valor de ejemplo. Si vas a exponer tu instancia de SearXNG fuera de tu equipo, sustitúyela por una clave propia.
- **Secretos.** Los tokens de los conectores se leen de variables de entorno (`${NOMBRE}` en `conectores.json`). Tus archivos `.env` y `conectores.json` están excluidos del control de versiones.
- **Datos.** `synaptocore_memory.db` contiene todo lo que el sistema ha aprendido. Guárdalo y compártelo como un dato privado.
