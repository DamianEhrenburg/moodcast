import html
from typing import Optional


def escape_html(text: Optional[str]) -> str:
    """Escapes HTML special characters for safe Telegram rendering."""
    if text is None:
        return ""
    return html.escape(str(text), quote=False)
