# ESPECTRAZO

**ESPECTRAZO** es un proyecto de procesamiento digital de imágenes que permite visualizar en tiempo real el recorrido del movimiento de una persona mediante una estela de siluetas.

A través de una cámara web, el sistema captura el movimiento, separa la silueta de la persona del fondo y superpone sus posiciones anteriores con diferentes niveles de transparencia, creando un efecto visual de espectro o rastro de movimiento.

## Objetivo

Desarrollar un sistema de procesamiento de imágenes en tiempo real capaz de detectar y segmentar la silueta de una persona en movimiento, acumulando sus posiciones recientes para generar un efecto de estela visual.

## Tecnologías utilizadas

- **Python 3:** lenguaje principal de desarrollo.
- **OpenCV:** captura de video, detección de movimiento y procesamiento de imágenes.
- **NumPy:** manipulación y procesamiento de imágenes como matrices.
- **Collections.deque:** almacenamiento temporal de las siluetas anteriores.
- **Cámara web:** dispositivo de captura de video.

## Funcionamiento

El sistema procesa cada fotograma mediante las siguientes etapas:

1. **Captura:** obtiene video en tiempo real desde la cámara web.
2. **Preprocesamiento:** aplica filtros para reducir el ruido de la imagen.
3. **Detección de movimiento:** utiliza sustracción de fondo adaptativa (MOG2) para separar a la persona del entorno.
4. **Segmentación:** genera una silueta binaria mediante umbralización y operaciones morfológicas.
5. **Acumulación:** almacena y superpone las siluetas anteriores con transparencia decreciente.
6. **Visualización:** muestra la silueta actual junto con la estela de movimiento.

## Características principales

- Captura de video en tiempo real.
- Segmentación de la silueta completa de una persona.
- Eliminación de ruido mediante operaciones morfológicas.
- Generación de estelas de movimiento.
- Personalización de la longitud, el color y la intensidad de la estela.
- Posibilidad de guardar el resultado en video.

## Resultado esperado

Una experiencia interactiva en la que el usuario puede observar cómo sus movimientos se transforman en una composición visual de siluetas superpuestas, permitiendo explorar diferentes formas y recorridos corporales.

## Autoras

- Angeles Anchico Quiñones
- Valeria Garcia Perez

**Universidad Autónoma de Occidente**  
Asignatura: Procesamiento Digital de Imágenes  
Docente: Nicolás Llanos Neuta  
2026
git remote add origin https://github.com/USUARIO/espectrazo.git
git push -u origin main
```
Guarda también el notebook en `notebooks/` (en Colab: Archivo → Guardar una copia en GitHub).

## Estructura
- `app.py` servidor Flask, cámara y endpoints · `pipeline.py` silueta, estela y CNN · `templates/index.html` interfaz
