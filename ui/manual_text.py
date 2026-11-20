"""Content of the user manual shown inside the application, without Qt.

The manual explains how to take the two photos so they can be compared
(camera position and alignment) and what the photos must meet (lighting,
resolution, format). Every figure is read from the settings the application
runs with, so the manual never disagrees with the checks the analysis applies.
The wording follows the rest of the interface: plain language, addressing the
user as "usted", without technical terms.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Optional

from config.settings import AppSettings
from controllers.pipeline_controller import LOW_OVERLAP_FRACTION
from models.comparison_report import ChangeType
from models.segmentation_mask import UNIDENTIFIED_LABEL
from models.session_history import SessionHistory
from ui.report_text import NO_CHANGES_MESSAGE

# Names of the buttons, actions and labels the manual tells the user to use,
# with their keyboard shortcut. The tests check them against the real window.
LOAD_FILE_BUTTON = "Cargar archivo..."
CAPTURE_BUTTON = "Capturar"
CAMERA_LABEL = "Cámara:"
ANALYZE_BUTTON = "Analizar diferencias"
EXAMPLES_MENU = "Cargar ejemplo"
ACTION_SHORTCUTS: dict[str, Optional[str]] = {
    "Nuevo análisis": "Ctrl+N",
    "Cargar fotos...": "Ctrl+O",
    "Capturar de cámara": None,
    "Guardar imagen...": "Ctrl+S",
    "Exportar...": "Ctrl+E",
    "Historial...": "Ctrl+H",
    "Manual de usuario": "F1",
}


@dataclass(frozen=True)
class ManualSection:
    """One chapter of the manual; ``anchor`` links to it from the index."""

    anchor: str
    title: str
    paragraphs: tuple[str, ...] = ()
    bullets: tuple[str, ...] = ()


def build_manual(settings: Optional[AppSettings] = None) -> tuple[ManualSection, ...]:
    """The eight sections of the manual, with the figures of ``settings`` (defaults when ``None``)."""
    s = settings or AppSettings()
    return (
        _purpose(),
        _first_steps(),
        _taking_photos(),
        _photo_requirements(s),
        _reading_results(),
        _history(),
        _messages(s),
        _limitations(),
    )


def manual_html(sections: tuple[ManualSection, ...]) -> str:
    """Simple HTML for a rich-text viewer: an index of internal links and one anchor per section.

    Colors and fonts are left to the viewer's style sheet.
    """
    index = "".join(
        f'<li><a href="#{html.escape(section.anchor)}">{html.escape(section.title)}</a></li>'
        for section in sections
    )
    parts = ["<h1>Manual de usuario</h1>", f"<p>Contenido:</p><ul>{index}</ul>"]
    for section in sections:
        parts.append(f'<a name="{html.escape(section.anchor)}"></a><h2>{html.escape(section.title)}</h2>')
        parts.extend(f"<p>{html.escape(paragraph)}</p>" for paragraph in section.paragraphs)
        if section.bullets:
            items = "".join(f"<li>{html.escape(bullet)}</li>" for bullet in section.bullets)
            parts.append(f"<ul>{items}</ul>")
    return "<html><body>" + "".join(parts) + "</body></html>"


def max_history_entries() -> int:
    """How many comparisons the session history keeps, as the window creates it."""
    return SessionHistory().max_entries


# ------------------------------------------------------------------ sections


def _purpose() -> ManualSection:
    return ManualSection(
        anchor="para-que-sirve",
        title="Para qué sirve",
        paragraphs=(
            "Esta aplicación compara dos fotografías del mismo espacio interior, tomadas en momentos "
            "distintos, e indica qué objetos se retiraron, cuáles se movieron (y cuántos centímetros) "
            "y cuáles aparecieron.",
            "Las fotos se procesan en este equipo y no se envían a internet.",
        ),
    )


def _first_steps() -> ManualSection:
    return ManualSection(
        anchor="primeros-pasos",
        title="Primeros pasos",
        paragraphs=(
            f"Para probar la aplicación sin fotos propias, abra el menú Archivo y elija «{EXAMPLES_MENU}»: "
            "carga con un clic un par de fotos de ejemplo, «Antes» y «Después», listo para analizar.",
            "Un análisis se hace en tres pasos:",
        ),
        bullets=(
            f"Elija las dos fotos. En el panel «Antes» pulse «{LOAD_FILE_BUTTON}» para abrir una foto "
            f"guardada, o «{CAPTURE_BUTTON}» para tomarla con la cámara indicada en «{CAMERA_LABEL}». "
            "Haga lo mismo en el panel «Después». También puede usar "
            f"{_action('Cargar fotos...')} en el menú Archivo: si elige dos fotos, la primera se toma "
            "como «Antes» y la segunda como «Después».",
            "Revise el estado de cada foto. Cada panel indica si la foto es válida o qué le falta; "
            "una foto con error debe reemplazarse.",
            f"Pulse «{ANALYZE_BUTTON}». El botón se habilita cuando las dos fotos son válidas. Una barra "
            "muestra el avance; la primera vez tarda unos segundos más mientras el sistema se prepara.",
            "Lea el resultado: la imagen con los cambios marcados y una tarjeta por objeto. "
            f"{_action('Exportar...')} guarda la imagen y un resumen; {_action('Guardar imagen...')} "
            "guarda solo la imagen.",
            f"Para comparar otro par, use {_action('Nuevo análisis')}. Los análisis anteriores siguen "
            f"disponibles en {_action('Historial...')}, y este manual se abre con "
            f"{_action('Manual de usuario')}.",
        ),
    )


def _taking_photos() -> ManualSection:
    return ManualSection(
        anchor="como-tomar-las-fotos",
        title="Cómo tomar las fotos",
        paragraphs=(
            "La aplicación compara las dos fotos punto por punto, así que el resultado depende sobre "
            "todo de que ambas muestren la escena desde el mismo lugar. Siga estas pautas:",
        ),
        bullets=(
            "Mantenga la cámara fija entre las dos tomas, por ejemplo sobre un trípode o apoyada en un "
            "mueble, o vuelva exactamente al mismo punto y con la misma orientación. Así cada parte de "
            "una foto coincide con la misma parte de la otra.",
            "Encuadre la zona que le interesa en las dos tomas y procure que ambas muestren la mayor "
            "parte de la misma escena. Los cambios solo se buscan en la zona que aparece en las dos "
            f"fotos; si esa zona es menor que el {LOW_OVERLAP_FRACTION * 100:.0f} % de la escena, "
            "la aplicación lo avisa.",
            "Incluya en el encuadre detalles fijos y reconocibles, como muebles, esquinas o marcos. "
            "La aplicación los usa para hacer coincidir las dos fotos; con muy pocos detalles en "
            "común no puede alinearlas.",
            "Tome las dos fotos con una iluminación parecida. La comparación también se fija en los "
            "colores, y un cambio de luz puede alterar tanto los colores como las distancias "
            "calculadas.",
            "Entre una toma y otra, cambie solo lo que quiere detectar. La aplicación informa cualquier "
            "objeto que se haya retirado, movido o agregado.",
            "Tome primero la foto «Antes» y después la foto «Después», y cárguelas en ese orden. La "
            "primera es la referencia: si se invierten, un objeto retirado aparecerá como nuevo.",
            "Use la misma cámara, la misma resolución y el mismo zoom en las dos tomas. Las distancias "
            "en centímetros dependen de las características de la cámara con que se tomó la foto.",
        ),
    )


def _photo_requirements(s: AppSettings) -> ManualSection:
    extensions = ", ".join(ext.lstrip(".").upper() for ext in s.image.allowed_extensions)
    bullets = [
        f"Formato: {extensions}. Una foto en otro formato se rechaza.",
        f"Resolución mínima: {s.image.min_width}x{s.image.min_height} píxeles, en posición horizontal "
        "o vertical. Una foto más pequeña se rechaza. Al capturar con la cámara, la aplicación "
        f"solicita fotos de {s.capture.frame_width}x{s.capture.frame_height}.",
        "Color: fotos a color (RGB) de 8 bits por canal, como las de cualquier celular o cámara. "
        "Una foto en blanco y negro se rechaza.",
        "Iluminación: una foto se considera poco iluminada cuando su brillo medio es menor que "
        f"{s.image.low_light_threshold:.0f} en una escala de 0 (negro) a 255 (blanco). Se analiza "
        "igual, pero con un aviso de que el resultado puede no ser confiable: con poca luz, las "
        "distancias pueden desviarse de forma importante, como se observó en una escena de prueba.",
    ]
    if _indoor_check_active(s):
        bullets.append(
            "Espacio interior: las dos fotos deben mostrar un espacio interior. Se admite hasta un "
            f"{s.segmentation.max_outdoor_percent:.0f} % de elementos de exterior, como el cielo o un "
            "árbol visto por una ventana; si una foto lo supera, el análisis se detiene e indica qué "
            "foto reemplazar. Esta comprobación ocurre al analizar, no al cargar la foto."
        )
    bullets.append(
        "Tamaño y tiempo: a mayor resolución, más tarda el análisis, sobre todo en un equipo sin "
        "tarjeta gráfica compatible."
    )
    return ManualSection(
        anchor="requisitos-de-las-fotos",
        title="Requisitos de las fotos",
        paragraphs=(
            "Cada foto se revisa al cargarla. Si no cumple el formato, la resolución o el color, se "
            "rechaza y el panel indica el motivo; si tiene poca luz, se acepta con un aviso.",
        ),
        bullets=tuple(bullets),
    )


def _reading_results() -> ManualSection:
    return ManualSection(
        anchor="como-leer-el-resultado",
        title="Cómo leer el resultado",
        paragraphs=(
            "Cada objeto que cambió tiene un número, que aparece en la imagen y en su tarjeta, junto "
            "con el tipo de cambio:",
        ),
        bullets=(
            f"{ChangeType.DISPLACED.title}: el objeto cambió de lugar. Su posición anterior se marca en "
            "naranja y la actual en verde, unidas por una flecha, y la tarjeta indica cuántos "
            "centímetros se movió.",
            f"{ChangeType.REMOVED.title}: el objeto ya no está. El lugar donde estaba se marca en rojo.",
            f"{ChangeType.ADDED.title}: el objeto no estaba en la foto «Antes». Se marca en azul.",
            "La distancia en centímetros es una estimación. Debajo de las tarjetas, una nota indica "
            "con qué datos de la cámara se calculó: la calibración de la cámara, los datos que la "
            "cámara guardó en la foto o, si la foto no los trae, una estimación estándar, que puede "
            "variar unos centímetros.",
            f"Si no hubo cambios, aparece el mensaje «{NO_CHANGES_MESSAGE}».",
            "Los avisos, como el de poca luz, aparecen en amarillo encima de las tarjetas. Puede "
            "acercar o alejar la imagen con la rueda del ratón y moverla arrastrándola.",
        ),
    )


def _history() -> ManualSection:
    return ManualSection(
        anchor="historial-de-la-sesion",
        title="Historial de la sesión",
        bullets=(
            f"Ábralo con {_action('Historial...')} en el menú Archivo o en la barra de herramientas.",
            "Muestra los análisis completados en esta sesión, del más reciente al más antiguo; al "
            "elegir uno se ve su resultado completo, que también puede exportarse desde ahí.",
            f"Guarda como máximo los {max_history_entries()} análisis más recientes. Los análisis que "
            "no terminaron, por ejemplo por una foto rechazada, no se guardan.",
            "El historial se pierde al cerrar la aplicación: no se guarda en el equipo.",
        ),
    )


def _messages(s: AppSettings) -> ManualSection:
    bullets = [
        "Foto poco iluminada: tome la foto con más luz, parecida en las dos tomas. Si la usa así, "
        "revise el resultado con cuidado.",
        f"Resolución insuficiente: use una foto de al menos {s.image.min_width}x{s.image.min_height} "
        "píxeles, por ejemplo la original de la cámara y no una copia reducida.",
        "Formato no compatible: guarde la foto en uno de los formatos admitidos (por ejemplo JPG o "
        "PNG) y cárguela de nuevo.",
        "Archivo dañado o que no se puede abrir: vuelva a copiar la foto desde el equipo o la cámara "
        "de origen.",
    ]
    if _indoor_check_active(s):
        bullets.append(
            "La foto no parece un espacio interior: reemplace la foto indicada por una tomada dentro "
            "de una habitación o área de trabajo."
        )
    bullets += [
        "Las fotos tienen pocos puntos en común o no se pudieron alinear: el resultado aparece con un "
        "aviso amarillo y la comparación se hace suponiendo que la cámara no se movió. Tome la "
        "segunda foto desde una posición y con un encuadre más parecidos a los de la primera.",
        "Solo una parte de la escena aparece en las dos fotos: los cambios fuera de esa parte no se "
        "detectan. Repita la segunda foto con el mismo encuadre que la primera.",
        "El equipo no tiene una tarjeta gráfica compatible: el análisis se hace igual, pero tarda "
        "más.",
        f"La cámara no está disponible o no entrega imagen: elija otro número en «{CAMERA_LABEL}», "
        "compruebe que esté conectada y cierre otras aplicaciones que la estén usando.",
    ]
    return ManualSection(
        anchor="avisos-y-errores",
        title="Avisos y errores frecuentes",
        paragraphs=("Qué hacer en cada caso:",),
        bullets=tuple(bullets),
    )


def _limitations() -> ManualSection:
    return ManualSection(
        anchor="limitaciones",
        title="Limitaciones",
        bullets=(
            "Está pensada solo para espacios interiores.",
            "Los nombres de los objetos se limitan a las categorías que el sistema conoce. Un objeto "
            "pequeño puede recibir el nombre de algo parecido (por ejemplo, una taza puede aparecer "
            "como «vaso» o «bolso», y un cubo como «caja»), y un objeto que no reconoce aparece como "
            f"«{UNIDENTIFIED_LABEL}».",
            "Las distancias son aproximadas, sobre todo cuando la foto no trae los datos de la cámara.",
            "Se ha probado sobre todo con la cámara fija; con la cámara en la mano la comparación se ha "
            "probado menos.",
        ),
    )


# ------------------------------------------------------------------- helpers


def _action(name: str) -> str:
    """Action name with its shortcut, e.g. «Exportar...» (Ctrl+E)."""
    shortcut = ACTION_SHORTCUTS[name]
    return f"«{name}» ({shortcut})" if shortcut else f"«{name}»"


def _indoor_check_active(s: AppSettings) -> bool:
    # The indoor check needs the object recognition, so it does not run when that is disabled.
    return s.segmentation.check_indoor_scene and s.segmentation.enabled
