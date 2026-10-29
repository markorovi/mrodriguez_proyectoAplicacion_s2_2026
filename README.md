# Detección de cambios en escenas mediante profundidad monocular

Aplicación de escritorio en Python que detecta cambios entre dos fotografías RGB de un
mismo espacio interior, tomadas en momentos distintos ("antes" y "después") con una cámara
convencional (webcam o teléfono). No requiere sensores de profundidad dedicados (LiDAR,
RGB-D): la geometría de la escena se infiere con un modelo de visión basado en transformers.

## Descripción técnica

El sistema ejecuta un pipeline de seis etapas:

1. **Captura**: carga de imágenes desde disco o captura desde una cámara RGB conectada, con
   validación de formato, resolución, canales e iluminación.
2. **Estimación de profundidad**: un mapa de profundidad **métrica** (en metros) por imagen
   con [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), variante
   *Metric Indoor*, ejecutado con PyTorch en GPU (CUDA) o, si no hay GPU, en CPU.
3. **Comparación espacial**: alineación por homografía (características + RANSAC),
   normalización robusta de escala y extracción de regiones de cambio.
4. **Clasificación**: cada cambio se clasifica como *desaparición* o *desplazamiento*; en
   este caso se calcula la distancia en centímetros por retroproyección con los parámetros
   intrínsecos de la cámara.
5. **Segmentación semántica**: categoría del objeto con SegFormer, restringida a clases de
   interiores.
6. **Reporte**: resultado en lenguaje no técnico con la región del cambio resaltada.

Todo el procesamiento es local; ninguna imagen se envía a servicios externos.

### Arquitectura

Arquitectura en capas con paquetes independientes. Solo la capa de orquestación invoca a los
servicios, de modo que cada modelo puede sustituirse sin modificar los demás módulos.

| Paquete | Responsabilidad |
|---|---|
| `config/` | Configuración tipada e inmutable (Pydantic): criterios de imagen, cámara, modelo e intrínsecos `fx, fy, cx, cy`. |
| `models/` | Estructuras de datos del dominio (`ImageData`, `DepthMap`, `ValidationReport`, `AcquiredImage`) y jerarquía de excepciones. Caché local de pesos en `models/weights/`. |
| `services/capture/` | Fuentes de imagen (`ImageSource`, implementada por `FileImageLoader` y `CameraCapture`) y `ImageAcquisitionService`, que entrega imágenes ya validadas. |
| `services/inference/` | `BaseDepthEstimator` (contrato común), `DepthAnythingV2Estimator`, `DeviceManager` (GPU/CPU), cargador de modelos de Hugging Face y `DepthLatencyBenchmark`. |
| `controllers/` | Orquestación del pipeline (fachada única para la interfaz). |
| `utils/` | `ImageValidator`, estadísticas fotométricas, coloreado de mapas de profundidad y generador de escenas sintéticas. |
| `ui/` | Interfaz gráfica de escritorio (PySide6): diálogo de selección de imágenes. |
| `scripts/` | Utilidades de línea de comandos (medición de latencia). |
| `tests/` | Pruebas unitarias y de integración con `pytest`. |
| `data/samples/` | Escenas interiores sintéticas para demostración y pruebas sin conexión. |

Patrones aplicados:

- **Estrategia**: `BaseDepthEstimator` define `estimate(image) -> DepthMap`; cualquier modelo
  de profundidad se integra implementando `_predict`, `load` y `unload`.
- **Método plantilla**: `estimate` valida la entrada, carga el modelo de forma perezosa, mide el
  tiempo, verifica la resolución de salida y traduce fallos de ejecución (p. ej. memoria de GPU
  insuficiente) a `DepthEstimationError`.
- **Adaptador**: `DepthAnythingV2Estimator` adapta el procesador y el modelo de Hugging Face a
  la interfaz uniforme del estimador.
- **Inyección de dependencias**: el dispositivo de video, el selector de hardware y el cargador
  de modelos son reemplazables, lo que permite probar cada módulo sin hardware ni descargas.

## Estado actual

**Avance 1 completado: infraestructura y estimación de profundidad.**

| Componente | Estado |
|---|---|
| Estructura modular, configuración, modelos de dominio | Completado |
| Carga desde disco (rutas con caracteres no ASCII, orientación EXIF, archivos dañados o truncados) | Completado |
| Captura desde cámara RGB (verificada con webcam física) | Completado |
| Validación: formato, resolución mínima 640x480, RGB de 8 bits, advertencia de baja iluminación | Completado |
| Diálogo gráfico de selección de imágenes "Antes" / "Después" con vista previa y estado | Completado |
| Estimación de profundidad métrica con Depth Anything V2 (GPU con precisión mixta / CPU) | Completado, verificada en GPU NVIDIA |
| Pruebas | 115 unitarias + 9 de integración |
| Alineación espacial, comparación y clasificación | Pendiente |
| Metrología 3D (distancia en cm) | Pendiente |
| Segmentación semántica (SegFormer) | Pendiente |
| Orquestador, reporte y ventana principal | Pendiente |

### Rendimiento medido

Variante *Small*, imagen 640x480, `python -m scripts.benchmark_depth`:

| Hardware | Precisión | Inferencia estable por imagen | Profundidad por par | Carga del modelo |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GB) | float16 (mixta) | 0.024 s | 0.048 s | 5.6-7.4 s |
| NVIDIA GeForce RTX 3070 (8 GB) | float32 | 0.039 s | 0.078 s | 5.6 s |
| CPU | float32 | 0.86 s | 1.72 s | 5.8 s |

La primera inferencia tras cargar el modelo tarda más (~0.7 s en GPU) por la inicialización de
CUDA. La carga del modelo es el costo dominante, por lo que conviene mantenerlo en memoria
entre análisis. En float16 la diferencia con float32 en CPU tiene una mediana menor al 2 %.

## Requisitos de las imágenes de entrada

| Criterio | Valor | Efecto si no se cumple |
|---|---|---|
| Formato | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` | Se rechaza |
| Resolución mínima | 640x480 (cualquier orientación) | Se rechaza |
| Color | RGB, 8 bits por canal (RGBA y paleta se convierten a RGB) | Se rechaza |
| Iluminación | Luminancia media >= 40 (escala 0-255) | Se procesa con advertencia |

Para obtener resultados confiables, tome ambas fotografías desde la misma posición y con la
misma orientación de cámara, con iluminación similar.

## Instalación

Requiere Python 3.10 o superior (verificado con 3.13).

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Aceleración por GPU (opcional)

En Windows, `pip` instala por defecto la versión de PyTorch solo para CPU. Con una GPU NVIDIA
cuyo controlador soporte CUDA 13.0 o superior (verifíquelo con `nvidia-smi`), instale además:

```bash
pip install -r requirements-cuda.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Debe imprimir: True NVIDIA GeForce ...
```

Para controladores más antiguos, use el comando que indica el selector de
<https://pytorch.org/get-started/locally/> para su versión de CUDA.

Sin GPU la aplicación funciona igual en CPU y avisa que el proceso puede tardar más.

### Pesos del modelo

La primera ejecución descarga los pesos (~100 MB para la variante *Small*) en
`models/weights/`. Las siguientes ejecuciones los reutilizan; con `local_files_only=True` en
la configuración no se accede a la red.

Revise la licencia de cada checkpoint en su ficha de Hugging Face antes de un uso comercial:
la variante *Small* se distribuye bajo Apache-2.0 y las variantes *Base* y *Large* bajo
CC-BY-NC-4.0.

## Diálogo de selección de imágenes

```bash
python -m ui                     # con la configuración por defecto
python -m ui settings.json       # con un archivo de configuración
```

El diálogo tiene un panel "Antes" y otro "Después". Cada uno permite **Cargar archivo...** o
**Capturar** desde la cámara indicada en el selector "Cámara". Cada imagen se valida al
cargarla y muestra su miniatura con un estado:

| Estado | Significado | ¿Permite continuar? |
|---|---|---|
| VÁLIDA | Cumple todos los criterios | Sí |
| ADVERTENCIA LUZ | Imagen poco iluminada; el resultado puede no ser confiable | Sí |
| ERROR RESOLUCIÓN | Menor que 640x480 | No |
| ERROR COLOR | No es RGB de 8 bits (p. ej. escala de grises) | No |
| ERROR FORMATO / ERROR ARCHIVO | Extensión no admitida, archivo dañado o inexistente | No |
| ERROR CÁMARA | La cámara no está disponible o no entregó imagen | No |

"Continuar" se habilita cuando ambas imágenes son utilizables; al aceptar, el diálogo imprime
el resumen del par seleccionado.

Si una computadora tiene varias cámaras (por ejemplo, una cámara infrarroja o virtual además de
la webcam), la numeración depende del sistema operativo. Pruebe con 0, 1, 2... o fije el valor en
`capture.camera_index` del archivo de configuración.

## Pruebas

```bash
# Pruebas unitarias (rápidas, sin red, GPU ni cámara; la interfaz se prueba sin ventana)
pytest

# Pruebas de integración con el modelo real (descargan los pesos la primera vez)
RUN_INTEGRATION=1 pytest -m integration -v -s           # bash
$env:RUN_INTEGRATION=1; pytest -m integration -v -s     # PowerShell
```

Las pruebas de integración de hardware ([tests/test_hardware_integration.py](tests/test_hardware_integration.py))
verifican que la inferencia corre en la GPU, miden la latencia, comparan float16 en GPU con
float32 en CPU, capturan desde la webcam y guardan el mapa de profundidad coloreado en
`data/samples/depth_output_sample.png`. Las que no encuentran su hardware o sus datos se omiten:

- **Fotos reales**: copie un par de fotos del mismo espacio como
  `data/samples/real_before.jpg` y `data/samples/real_after.jpg` (cualquier formato admitido).
- **Webcam**: se prueban los índices 0-3; para fijar uno, defina `CAMERA_INDEX`
  (`$env:CAMERA_INDEX=1` en PowerShell).

### Medición de latencia

```bash
python -m scripts.benchmark_depth                          # GPU si está disponible
python -m scripts.benchmark_depth --device cpu             # forzar CPU
python -m scripts.benchmark_depth --full-precision         # GPU en float32
python -m scripts.benchmark_depth --variant base --repetitions 20 --image ruta/a/foto.jpg
```

## Uso programático

```python
from config import load_settings
from services.capture import ImageAcquisitionService
from services.inference import create_depth_estimator

settings = load_settings()                      # o load_settings("settings.json")
acquisition = ImageAcquisitionService()

before = acquisition.from_file("data/samples/scene_before.png")
for warning in before.warnings:
    print("Advertencia:", warning)

with create_depth_estimator(settings.depth) as estimator:
    if estimator.compute_device.notice:
        print(estimator.compute_device.notice)
    depth = estimator.estimate(before.image)

print(depth.shape, f"{depth.min_depth:.2f}-{depth.max_depth:.2f} m")
```

### Configuración

Todos los parámetros tienen valores por defecto. Un archivo JSON puede sobrescribir solo los
necesarios:

```json
{
  "depth": { "variant": "base", "device": "cuda", "half_precision": true },
  "camera": { "fx": 1450.0, "fy": 1450.0, "cx": 960.0, "cy": 540.0,
              "reference_width": 1920, "reference_height": 1080 },
  "capture": { "camera_index": 1 }
}
```

Los intrínsecos por defecto corresponden a una cámara típica de 640x480 y se reescalan con
`CameraIntrinsics.scaled_to(width, height)`. Para mediciones en centímetros precisas, use los
valores de una calibración de su cámara.

## Estructura del proyecto

```
|-- config/              Configuración y parámetros (Pydantic)
|-- models/              Estructuras de datos del dominio y caché de pesos
|-- services/
|   |-- capture/         Carga desde disco y captura desde cámara
|   `-- inference/       Estimación de profundidad (Depth Anything V2) y selección de hardware
|-- controllers/         Orquestación del pipeline
|-- utils/               Validación de imágenes, visualización y utilidades numéricas
|-- ui/                  Interfaz gráfica (diálogo de selección de imágenes)
|-- scripts/             Medición de latencia
|-- tests/               Pruebas con pytest
|-- data/samples/        Imágenes de ejemplo y mapa de profundidad de muestra
|-- pyproject.toml       Configuración de pytest
|-- requirements.txt
`-- requirements-cuda.txt  PyTorch con soporte CUDA (opcional)
```

## Próximos pasos

1. **Alineación espacial**: homografía entre "antes" y "después" (ORB + RANSAC), deformación de
   la imagen y del mapa de profundidad, y normalización robusta de escala y desplazamiento
   sobre los píxeles estáticos.
2. **Detección y clasificación de cambios**: diferencia de profundidad con umbral, morfología y
   componentes conexas; separación en regiones vaciadas y ocupadas; emparejamiento por
   categoría y similitud de apariencia para distinguir desaparición de desplazamiento.
3. **Metrología 3D**: retroproyección de los centroides con la mediana de profundidad y los
   intrínsecos `fx, fy, cx, cy`; distancia euclidiana en centímetros con umbral mínimo
   reportable.
4. **Segmentación semántica**: integración de SegFormer con restricción a categorías de
   interiores y liberación secuencial de modelos para limitar el uso de VRAM.
5. **Orquestación, reporte e interfaz**: `PipelineController`, resaltado de regiones, historial
   de la sesión y ventana principal PySide6 que integra el diálogo de selección, con flujo de
   tres interacciones.
