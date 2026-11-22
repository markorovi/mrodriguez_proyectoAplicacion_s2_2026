"""Technical terms that must not appear in any text shown to the user (RNF-001).

Shared by every test that checks user-facing wording, so the list lives in one
place. Terms are compared ignoring case and accents (see :func:`jargon_found`).
"""

from __future__ import annotations

import unicodedata

JARGON = (
    "homografía",
    "ransac",
    "argmax",
    "pytorch",
    "cuda",
    "sift",
    "tensor",
    "segment",
    # Model names, image-format details and processing concepts the user does not need.
    "depth-anything",
    "segformer",
    "nvidia/",
    "rgb",
    "8 bits",
    "canales",
    "correspondencias",
    "geométricamente",
    "gpu",
    "cpu",
)


def _fold(text: str) -> str:
    """Lowercase ``text`` without accents, so "Geométricamente" and "geometricamente" compare equal."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def jargon_found(text: str) -> list[str]:
    """Terms of :data:`JARGON` present in ``text``, ignoring case and accents."""
    folded = _fold(text)
    return [term for term in JARGON if _fold(term) in folded]
