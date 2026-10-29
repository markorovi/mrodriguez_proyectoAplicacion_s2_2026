# Imágenes de ejemplo

Escenas interiores sintéticas de 640x480 generadas con `utils/synthetic_scene.py`
(semilla 7). Sirven para demostraciones y pruebas sin conexión.

| Archivo | Contenido |
|---|---|
| `scene_before.png` | Habitación con una caja roja a la izquierda (estado "antes"). |
| `scene_after_moved.png` | La misma caja desplazada hacia la derecha. |
| `scene_after_removed.png` | La caja ya no está. |
| `scene_low_light.png` | `scene_before.png` con un 12 % de brillo; dispara la advertencia de baja iluminación. |
| `depth_output_sample.png` | Mapa de profundidad coloreado (cerca = claro) generado por las pruebas de integración en GPU; usa `real_before.*` si existe y, si no, `scene_before.png`. |

Para probar con fotos reales, copie aquí un par como `real_before.<ext>` y `real_after.<ext>`.

Para regenerarlas:

```python
from PIL import Image
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

builder = SyntheticSceneBuilder(640, 480, seed=7)
Image.fromarray(builder.render([SceneObject(center_x=0.35)])).save("data/samples/scene_before.png")
```
