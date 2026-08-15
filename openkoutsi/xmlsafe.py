"""Streaming XML reading for activity files that arrive from strangers.

GPX and TCX are the two formats openkoutsi ingests that are XML, and both reach
the parser straight from an upload — including from inside a zip a user
downloaded from somewhere else. Two properties matter here and neither is the
standard library's default posture:

* **No document type declaration, ever.** Every classic XML attack on a parser
  needs a DTD: external entities to read files off the server or make it fetch
  URLs, and internal entity nesting for the exponential expansion known as the
  billion laughs. ``xml.etree.ElementTree`` does not resolve *external* entities,
  but it does expand internal ones, and neither hardening depends on which
  parser accelerator is compiled in if the declaration is simply refused before
  parsing starts. :func:`iter_elements` rejects any document carrying one.
* **Bounded memory.** A three-hour ride is tens of thousands of track points.
  Building the whole tree to then walk it costs an order of magnitude more than
  the samples are worth, so this yields one element at a time and unlinks each
  from its parent as it goes — a file's peak cost is a single track point, not
  the file.

Namespaces are stripped rather than matched: GPX 1.0, GPX 1.1, TCX and every
vendor extension disagree about the namespace URI and agree about the tag name,
and no activity file needs two different ``hr`` elements told apart.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Union

Fileish = Union[str, Path, bytes, bytearray, IO[bytes]]


class XmlSafetyError(ValueError):
    """The document is structurally hostile — a DTD, or not XML at all."""


_WS = b" \t\r\n"
# Prolog constructs that may legitimately precede the root element. A document
# type declaration is deliberately not among them.
_PROLOG_OK = (b"<?", b"<!--")


def read_bytes(fileish: Fileish) -> bytes:
    """Whatever the caller had, as bytes.

    Activity files are read whole rather than streamed from disk because the
    same parser has to serve a path, an upload buffer and an entry decompressed
    out of a zip. Size is bounded upstream: the upload endpoint caps the request
    and the archive walker caps each entry's *uncompressed* size, so nothing
    here can be handed a file bigger than those limits allow.
    """
    if isinstance(fileish, (bytes, bytearray)):
        return bytes(fileish)
    if isinstance(fileish, (str, Path)):
        return Path(fileish).read_bytes()
    data = fileish.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def reject_doctype(data: bytes) -> None:
    """Raise if the document declares a DTD.

    Walks the prolog rather than searching the whole file for ``<!DOCTYPE``:
    the string can legitimately appear inside a track name, and a declaration
    can only appear before the root element, so the precise check is also the
    cheap one.
    """
    i = 0
    n = len(data)
    # A UTF-8/UTF-16 BOM is the parser's problem, not ours, but it has to be
    # stepped over to find the first markup.
    for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
        if data.startswith(bom):
            i = len(bom)
            break

    while i < n:
        while i < n and data[i] in _WS:
            i += 1
        if i >= n:
            return
        if data[i] != 0x3C:  # '<'
            # Text before the root element: not well-formed XML. Let the real
            # parser produce the error message; it words it better than we can.
            return
        rest = data[i:]
        if rest.startswith(b"<!DOCTYPE"):
            raise XmlSafetyError(
                "XML document type declarations are not accepted in activity files"
            )
        if not rest.startswith(_PROLOG_OK):
            return  # Root element start tag — the prolog is over.
        close = b"-->" if rest.startswith(b"<!--") else b"?>"
        end = data.find(close, i)
        if end < 0:
            return  # Truncated prolog; the parser will say so.
        i = end + len(close)


def local_name(tag: str) -> str:
    """``{http://www.topografix.com/GPX/1/1}trkpt`` → ``trkpt``."""
    if tag and tag[0] == "{":
        return tag.rsplit("}", 1)[-1]
    return tag


def iter_elements(data: bytes, wanted: frozenset[str]) -> Iterator[ET.Element]:
    """Yield completed elements whose local name is in ``wanted``, in document order.

    Each element is unlinked from the tree once the consumer has seen it, so the
    tree never grows past the elements still open. Consumers must therefore read
    what they need from an element before asking for the next one.
    """
    reject_doctype(data)

    stack: list[ET.Element] = []
    try:
        for event, elem in ET.iterparse(io.BytesIO(data), events=("start", "end")):
            if event == "start":
                stack.append(elem)
                continue
            stack.pop()
            if local_name(elem.tag) in wanted:
                yield elem
                elem.clear()
                if stack:
                    # Detach the husk, or a long ride accumulates one empty
                    # element per track point for the life of the parse.
                    parent = stack[-1]
                    try:
                        parent.remove(elem)
                    except ValueError:
                        pass
    except ET.ParseError as exc:
        raise XmlSafetyError(f"Malformed XML: {exc}") from exc


def root_tag(data: bytes, *, limit: int = 8192) -> str | None:
    """The document's root element name, without parsing it.

    Used to tell a GPX from a TCX when the filename does not (a zip entry called
    ``activity.xml``, a browser that guessed the extension). Reads only the head
    of the file.
    """
    match = re.search(rb"<\s*([A-Za-z_][\w.\-]*(?::[A-Za-z_][\w.\-]*)?)", data[:limit])
    if match is None:
        return None
    name = match.group(1).decode("ascii", errors="replace")
    return name.rsplit(":", 1)[-1]


def text_float(elem: ET.Element | None) -> float | None:
    """An element's text as a float, or ``None`` if it is missing or not a number."""
    if elem is None or elem.text is None:
        return None
    try:
        return float(elem.text.strip())
    except (TypeError, ValueError):
        return None
