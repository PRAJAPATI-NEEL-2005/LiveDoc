"""
LiveDoc text cleanup for drawn output.

Smart AI (Gemini) returns Markdown + LaTeX ($x^{(i)}$, \\hat{A}_{-i}, **bold** ...).
Drawing that verbatim puts raw markup into the PDF / 2D view. `plain_text()` turns it
into clean Unicode text: Greek letters and operators, real super/subscript characters,
combining accents (hat / bar), fractions as a/b, and Markdown emphasis removed.
"""

import re

_SUP = {
    **{c: s for c, s in zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")},
    **{c: s for c, s in zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")},
    **{c: s for c, s in zip("ABDEGHIJKLMNOPRTUVW", "ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ")},
    "*": "*", "′": "′",
}
_SUB = {
    **{c: s for c, s in zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")},
    **{c: s for c, s in zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")},
}

_EXTRA_SYMBOLS = {
    r"\mid": "|", r"\ne": "≠", r"\ell": "ℓ", r"\hbar": "ħ", r"\propto": "∝",
    r"\circ": "∘", r"\bullet": "•", r"\therefore": "∴", r"\because": "∵",
    r"\Longrightarrow": "⟹", r"\Longleftrightarrow": "⟺", r"\angle": "∠",
    r"\leqslant": "≤", r"\geqslant": "≥", r"\ll": "≪", r"\gg": "≫",
    r"\varphi": "φ", r"\varrho": "ρ", r"\varsigma": "ς", r"\hat": "", r"\mathbb": "",
    r"\lbrace": "{", r"\rbrace": "}", r"\langle": "⟨", r"\rangle": "⟩",
    r"\star": "⋆", r"\oplus": "⊕", r"\otimes": "⊗", r"\sqrt": "√",
}

_ACCENTS = {
    "hat": "\u0302", "widehat": "\u0302", "tilde": "\u0303", "widetilde": "\u0303",
    "vec": "\u20d7", "dot": "\u0307", "ddot": "\u0308", "bar": "\u0304",
}
_PASSTHROUGH = (
    "text|textrm|textsf|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|"
    "mathfrak|operatorname|boldsymbol|bm|mbox|displaystyle|underline|boxed"
)

_ENTITY_FIX = {
    "&emsp;": "  ", "&thinsp;": " ", "&frasl;": "/", "&nbsp;": " ",
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"
}


_MATH_TABLE = None


def _get_math_table():
    global _MATH_TABLE
    if _MATH_TABLE is None:
        try:
            from src.pdf_exporter import PDFExporter
            _MATH_TABLE = {**PDFExporter.MATH_REPLACEMENTS, **_EXTRA_SYMBOLS}
        except Exception:
            _MATH_TABLE = dict(_EXTRA_SYMBOLS)
    return _MATH_TABLE


def _script(content: str, table: dict, marker: str) -> str:
    content = content.strip()
    if content and all(ch in table for ch in content):
        return "".join(table[ch] for ch in content)
    return f"{marker}({content})" if len(content) > 1 else f"{marker}{content}"


def _wrap(s: str) -> str:
    s = s.strip()
    return s if re.fullmatch(r"[\w.\u0300-\u036f]+", s) else f"({s})"


def _latex_to_plain(s: str) -> str:
    s = s.replace("$$", " ").replace("$", "")
    s = s.replace("\\{", "\x01").replace("\\}", "\x02")
    s = re.sub(r"\\(?:left|right)\s*(?=[\(\)\[\]\|\\.<>/])", "", s)
    s = re.sub(r"\\(?:left|right)\b\s*", "", s)

    # Brace commands, innermost first.
    for _ in range(12):
        prev = s
        s = re.sub(r"\\(?:%s)\s*\{([^{}]*)\}" % _PASSTHROUGH, r"\1", s)
        s = re.sub(r"\\overline\s*\{([^{}]*)\}", lambda m: "".join(c + "\u0305" for c in m.group(1)), s)
        s = re.sub(
            r"\\(hat|widehat|tilde|widetilde|vec|dot|ddot|bar)\s*\{([^{}]*)\}",
            lambda m: (m.group(2)[:1] + _ACCENTS[m.group(1)] + m.group(2)[1:]) if m.group(2) else "",
            s,
        )
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda m: f"{_wrap(m.group(1))}/{_wrap(m.group(2))}", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: "√" + _wrap(m.group(1)), s)
        s = re.sub(r"\\x(?:right|left)arrow\s*\{[^{}]*\}", "→", s)
        s = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
        s = re.sub(r"_\s*\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
        if s == prev:
            break

    # Symbol / Greek commands by exact command name (no prefix collisions).
    table = _get_math_table()
    s = re.sub(r"\\([A-Za-z]+)", lambda m: table.get("\\" + m.group(1), m.group(0)), s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\\\", " ").replace("\\|", "‖")
    s = re.sub(r"\\([#%&_$])", r"\1", s)

    # Bare ^x / _x directly after a symbol (x^2, A_i, σ²) but never inside words (file_name).
    s = re.sub(r"(?<=[\w\)\]\u0300-\u036f])\^([0-9A-Za-z+\-])(?![A-Za-z0-9])", lambda m: _script(m.group(1), _SUP, "^"), s)
    s = re.sub(r"(?<=[\w\)\]\u0300-\u036f])_([0-9a-z])(?![A-Za-z0-9])", lambda m: _script(m.group(1), _SUB, "_"), s)

    s = re.sub(r"\\[A-Za-z]+", "", s)          # unknown leftover commands
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\x01", "{").replace("\x02", "}")
    return s


def plain_text(text) -> str:
    """
    Markdown + LaTeX -> clean printable Unicode text (single line/paragraph, zero markup).
    Removes Markdown emphasis (**bold**, *italic*, __bold__, `code`), LaTeX syntax,
    HTML entities, blockquote markers (>), and cleans stray symbols like && -> ∧ or &.
    """
    if text is None:
        return ""
    s = str(text)
    if not s.strip():
        return ""
    for ent, rep in _ENTITY_FIX.items():
        s = s.replace(ent, rep)

    # Convert LaTeX formulas if present
    if "$" in s or "\\" in s or re.search(r"[\^_]\{", s):
        try:
            s = _latex_to_plain(s)
        except Exception:
            pass

    # Post-LaTeX entity cleanup (e.g. &emsp; produced by \quad)
    for ent, rep in _ENTITY_FIX.items():
        s = s.replace(ent, rep)

    # Blockquote markers (> 📷 Figure -> 📷 Figure)
    s = re.sub(r"^\s*>\s*", "", s)

    # Headings hashes (# Title -> Title)
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s)

    # Bullet markers (* Item -> • Item, - Item -> • Item)
    s = re.sub(r"^\s*[\*\-\+]\s+(?=\S)", "• ", s)

    # Markdown emphasis / code / links / images
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<![\w*])__(.+?)__(?![\w*])", r"\1", s)
    s = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]*?\S)\*(?![\w*])", r"\1", s)
    s = re.sub(r"(?<![\w*])_(?=\S)([^_\n]*?\S)_(?![\w*])", r"\1", s)
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)

    # Clean logic / stray && (e.g. 'A && B' -> 'A ∧ B', or stray '&&' -> '∧')
    s = re.sub(r"(?<=\w)\s*&&\s*(?=\w)", " ∧ ", s)
    s = re.sub(r"\s*&&\s*", " ∧ ", s)

    # Cleanup any leftover unpaired markdown artifacts
    s = s.replace("**", "").replace("__", "").replace("~~", "").replace("`", "")

    # Cleanup stray lone asterisk at beginning of text
    s = re.sub(r"^\s*\*\s*", "• ", s)

    return re.sub(r"[ \t\r\n]+", " ", s).strip()


clean_display_text = plain_text

