"""Outbound body rendering.

Providers like Lettermint accept only ``html`` and ``text`` — no markdown field
and no server-side templating — so the backend renders both parts itself and
always sends both. This module owns that rendering: a small Jinja2-driven helper
that produces a minimal, **inline-styled** HTML body (Gmail/Outlook strip
``<style>`` blocks, so every rule is inline) plus a plain-text alternative from
the same content.

Feature code (e.g. signup verification, password reset) calls
:func:`render_transactional_email` with its copy and wraps the result in an
:class:`~backend.app.services.email.base.OutboundMessage`.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html.j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_transactional_email(
    *,
    title: str,
    intro: str,
    body_paragraphs: Sequence[str] = (),
    action_label: str | None = None,
    action_url: str | None = None,
    outro: str | None = None,
    footer: str = "You received this message from an openkoutsi instance.",
) -> tuple[str, str]:
    """Render a transactional email into ``(html, text)``.

    Both parts are produced from the same content so they stay in sync. Pass an
    ``action_label`` + ``action_url`` to render a call-to-action button (HTML)
    and a labelled link (text).
    """
    context = {
        "title": title,
        "intro": intro,
        "body_paragraphs": list(body_paragraphs),
        "action_label": action_label,
        "action_url": action_url,
        "outro": outro,
        "footer": footer,
    }
    env = _env()
    html = env.get_template("transactional.html.j2").render(context)
    # Collapse the blank lines the text template naturally accumulates so the
    # plain-text part stays tidy regardless of which optional blocks are present.
    text_raw = env.get_template("transactional.txt.j2").render(context)
    text = _collapse_blank_lines(text_raw)
    return html, text


def _collapse_blank_lines(text: str) -> str:
    """Trim trailing whitespace and collapse runs of blank lines into one."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped and lines and not lines[-1]:
            continue
        lines.append(stripped)
    return "\n".join(lines).strip() + "\n"
