# ESPECTRAZO

## 1. Descripción del proyecto

**ESPECTRAZO** es un proyecto de procesamiento digital de imágenes que busca representar en tiempo real el recorrido del movimiento corporal mediante una estela de siluetas.

El proyecto surge de la necesidad de visualizar el movimiento de una persona mientras ocurre, sin depender de técnicas tradicionales como la fotografía de larga exposición o la cronofotografía, que requieren equipos especializados y edición posterior.

La solución propuesta utiliza una cámara web para capturar el movimiento, separar la silueta de la persona del fondo y acumular sus posiciones anteriores con diferentes niveles de transparencia, generando un efecto visual de espectro.

### Objetivo general

Desarrollar un sistema de procesamiento digital de imágenes en tiempo real que permita segmentar la silueta de una persona en movimiento y acumular sus posiciones recientes para generar un efecto de estela visual.

### Objetivos específicos

- Implementar la captura de video y la segmentación de personas mediante sustracción de fondo adaptativa.
- Generar siluetas limpias mediante umbralización y operaciones morfológicas.
- Desarrollar un sistema de acumulación de siluetas con transparencia decreciente.
- Evaluar la calidad visual y el rendimiento del sistema en tiempo real.

## 2. Arquitectura del sistema

El sistema está compuesto por seis etapas principales:

1. **Captura de video:** obtiene los fotogramas mediante una cámara web utilizando OpenCV.
2. **Preprocesamiento:** aplica filtros para reducir el ruido y las variaciones de iluminación.
3. **Detección de movimiento:** utiliza el modelo MOG2 para separar el movimiento del fondo.
4. **Segmentación de silueta:** aplica umbralización y operaciones morfológicas para obtener una figura limpia.
5. **Acumulación de posiciones:** superpone las siluetas anteriores con transparencia decreciente.
6. **Visualización:** muestra la silueta actual y su estela en tiempo real.

El diagrama de arquitectura se incluirá en esta sección para representar el flujo de información y la conexión entre los componentes.

## 3. Metodología

El desarrollo del proyecto se organiza en seis fases:

- **Fase 1. Planeación:** definición del alcance, revisión de conceptos y preparación del entorno de desarrollo.
- **Fase 2. Adquisición y preprocesamiento:** implementación de la captura de video y reducción de ruido.
- **Fase 3. Detección de movimiento:** implementación de la sustracción de fondo adaptativa y comparación con la diferencia entre fotogramas.
- **Fase 4. Obtención de silueta:** aplicación de umbralización, apertura y cierre morfológico.
- **Fase 5. Acumulación y efecto espectro:** desarrollo de la estela mediante almacenamiento temporal y transparencia decreciente.
- **Fase 6. Integración y pruebas:** integración de los módulos, evaluación del rendimiento y documentación.

### Técnicas y modelos utilizados

- Sustracción de fondo adaptativa (MOG2).
- Umbralización.
- Filtro gaussiano.
- Apertura y cierre morfológico.
- Selección del componente conectado de mayor área.
- Composición de imágenes con transparencia.

## 4. Implementación

El proyecto utiliza las siguientes tecnologías:

- **Python 3:** lenguaje principal.
- **OpenCV:** captura y procesamiento de imágenes.
- **NumPy:** manipulación de imágenes mediante matrices.
- **Collections.deque:** almacenamiento temporal de las siluetas.
- **Visual Studio Code:** entorno de desarrollo.
- **Git y GitHub:** control de versiones.

En esta sección se documentará el código fuente, los scripts y las principales funcionalidades de cada módulo, incluyendo instrucciones para ejecutar el sistema.

## 5. Datos

ESPECTRAZO utiliza como entrada principal el video capturado directamente desde una cámara web.

No depende de un conjunto de datos externo para su funcionamiento, ya que procesa los fotogramas en tiempo real.

Para las pruebas se contemplan diferentes escenarios:

- Personas con distintos movimientos.
- Fondos lisos y fondos con objetos.
- Condiciones de iluminación alta, baja y cambiante.
- Movimientos lentos y rápidos.

Se incluirán ejemplos de los fotogramas utilizados y las máscaras de segmentación obtenidas durante las pruebas.

## 6. Interfaz gráfica de usuario

La interfaz está diseñada para que el usuario pueda interactuar fácilmente con el sistema.

Se contemplan dos pantallas principales:

**Pantalla de inicio y captura**
- Vista previa de la cámara web.
- Botón para iniciar la cámara.
- Botón para detener la captura.
- Indicaciones para ubicar al usuario frente a la cámara.

**Pantalla de resultados**
- Visualización de la silueta actual y su estela.
- Indicador de rendimiento en FPS.
- Control para ajustar la longitud de la estela.
- Selector de color.
- Control de intensidad.
- Opción para guardar el video.

Se incluirán los prototipos y capturas de pantalla de la interfaz gráfica.

## 7. Resultados

La evaluación del sistema contempla pruebas de calidad visual y rendimiento.

Las métricas propuestas son:

- **IoU y Dice:** para evaluar la precisión de la segmentación.
- **Componentes espurios y porcentaje de huecos:** para medir la limpieza de la silueta.
- **Estabilidad temporal:** para analizar las variaciones de la silueta entre fotogramas.
- **Escala Likert:** para evaluar la calidad visual de la estela.
- **FPS y latencia:** para medir el rendimiento del procesamiento en tiempo real.

Como metas iniciales se plantea alcanzar un IoU igual o superior a 0,85, una valoración promedio de al menos 4 sobre 5 y un rendimiento mínimo de 20 FPS a una resolución de 640 × 480.

Los resultados obtenidos, ejemplos de funcionamiento y análisis de las pruebas se documentarán en esta sección.

## 8. Conclusiones

En esta sección se presentarán las conclusiones del desarrollo de ESPECTRAZO, teniendo en cuenta el cumplimiento de los objetivos, la calidad de la segmentación, el comportamiento de la estela y el rendimiento del procesamiento en tiempo real.

También se identificarán las limitaciones encontradas durante las pruebas y las posibles mejoras para futuras versiones del proyecto.

---

**Desarrollado por:**

- Angeles Anchico Quiñones
- Valeria Garcia Perez

**Universidad Autónoma de Occidente**  
Asignatura: Procesamiento Digital de Imágenes  
Docente: Nicolás Llanos Neuta  
2026
