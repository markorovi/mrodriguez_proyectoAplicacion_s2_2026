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
| `comparison_output_sample.png` | Resultado del análisis completo de `scene_before.png` contra `scene_after_moved.png`, con la posición anterior, la actual y la distancia recorrida; lo generan las pruebas de integración del pipeline. |
| `ejemplo_01_a.jpg` a `ejemplo_03_b.jpg` | Tres pares de fotos reales de un espacio interior (1920x1080), tomadas por el autor del proyecto. Son copias de sus fotos con los metadatos eliminados (sin EXIF ni ubicación GPS); `_a` es la foto "antes" y `_b` la "después". |
| `examples.json` | Manifiesto de los ejemplos que ofrece la aplicación en **Archivo > Cargar ejemplo**: las tres escenas sintéticas y los tres pares `ejemplo_*`. |

Los nombres `ejemplo_*` no empiezan por `real_` a propósito: las pruebas de integración tratan los
pares `real_before*` / `real_after*` como fotos de validación, y los ejemplos no deben alterarlas.

Para probar con fotos reales, copie aquí cada par como `real_before<sufijo>.<ext>` y
`real_after<sufijo>.<ext>` (por ejemplo `real_before_2.jpg` y `real_after_2.jpg`). Las pruebas de
integración analizan todos los pares y guardan cada resultado anotado como
`real_comparison_output<sufijo>.png`.

El resultado esperado de cada par se declara en `real_pairs.json`; todos los campos son opcionales:

```json
{
  "real_2": {
    "changes": ["DESPLAZAMIENTO"],
    "label": "silla",
    "distance_cm": 45.0,
    "distance_label": "silla",
    "distance_tolerance": 0.3
  }
}
```

`changes` son los tipos de cambio esperados (`DESPLAZAMIENTO`, `RETIRADO / DESAPARICIÓN`,
`APARICIÓN`), `label` el nombre del objeto principal y `distance_cm` la distancia medida con cinta,
que se compara con una tolerancia relativa (30 % por defecto) contra el objeto desplazado llamado
`distance_label` (o el desplazado más grande si se omite).

Para regenerarlas:

```python
from PIL import Image
from utils.synthetic_scene import SceneObject, SyntheticSceneBuilder

builder = SyntheticSceneBuilder(640, 480, seed=7)
Image.fromarray(builder.render([SceneObject(center_x=0.35)])).save("data/samples/scene_before.png")
```
