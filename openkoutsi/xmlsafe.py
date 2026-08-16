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
import math
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


# Encodings this module cannot scan, because the prolog walk below compares
# bytes against ASCII literals and a wide encoding's `<!DOCTYPE` is not those
# bytes. Each is recognised either by its byte-order mark or, for the BOM-less
# case, by what `<?` looks like once padded with NULs.
#
# Refusing them is not a limitation worth engineering around: the XML spec
# requires a conforming processor to support UTF-16, but no device or exporter
# in the wild writes a UTF-16 GPX or TCX, and accepting one would mean carrying
# an encoding matrix through the one check that must not have holes in it.
_UNSCANNABLE_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "UTF-32 (little-endian)"),
    (b"\x00\x00\xfe\xff", "UTF-32 (big-endian)"),
    (b"\xff\xfe", "UTF-16 (little-endian)"),
    (b"\xfe\xff", "UTF-16 (big-endian)"),
    (b"\x3c\x00\x3f\x00", "UTF-16 (little-endian)"),
    (b"\x00\x3c\x00\x3f", "UTF-16 (big-endian)"),
    (b"\x4c\x6f\xa7\x94", "EBCDIC"),
)


def reject_doctype(data: bytes) -> None:
    """Raise if the document declares a DTD, or is in an encoding we cannot scan.

    Walks the prolog rather than searching the whole file for ``<!DOCTYPE``:
    the string can legitimately appear inside a track name, and a declaration
    can only appear before the root element, so the precise check is also the
    cheap one.

    The walk is byte-oriented, which is only sound for an ASCII-compatible
    encoding. A UTF-16 document spells its markup ``3C 00 21 00 44 00 …``, so
    every literal below silently fails to match and the scan reports a clean
    prolog on a document carrying a DTD — the guarantee has to be one cheap
    check with no holes, so such documents are refused outright rather than
    scanned through a decoder.
    """
    for prefix, encoding in _UNSCANNABLE_PREFIXES:
        if data.startswith(prefix):
            raise XmlSafetyError(
                f"Activity files must be ASCII-compatible (this one looks like {encoding})"
            )

    i = 0
    n = len(data)
    # A UTF-8 BOM is the parser's problem, not ours, but it has to be stepped
    # over to find the first markup.
    if data.startswith(b"\xef\xbb\xbf"):
        i = 3

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


def _detach(parent: ET.Element | None, elem: ET.Element) -> None:
    elem.clear()
    if parent is not None:
        try:
            parent.remove(elem)
        except ValueError:
            pass


def iter_elements(data: bytes, wanted: frozenset[str]) -> Iterator[ET.Element]:
    """Yield completed elements whose local name is in ``wanted``, in document order.

    Two kinds of element are unlinked as the parse goes, so the tree never grows
    past what is still needed:

    * A **wanted** element, once the consumer has seen it. Consumers must
      therefore read what they need from an element before asking for the next.
    * An **unwanted** element that closes while no wanted element is open. A
      wanted element's descendants have to survive until it is yielded — the GPX
      parser reads ``<name>`` and ``<type>`` off a closed ``<trk>``, and the TCX
      parser walks a ``<Trackpoint>``'s children — but everything outside them
      is finished with the moment it closes, and holding onto it was letting a
      file of elements this parser does not want cost several times its own size
      in resident tree.

    The bound is therefore "the open elements plus the subtree under the
    outermost wanted one", which for a GPX or TCX activity is a track point. A
    file with a wanted element wrapping the whole document is still held whole;
    no format read here is shaped that way.
    """
    reject_doctype(data)

    stack: list[ET.Element] = []
    # Depth of the outermost currently-open wanted element, or None. Its whole
    # subtree has to stay intact until it is yielded.
    wanted_open_at: int | None = None

    try:
        for event, elem in ET.iterparse(io.BytesIO(data), events=("start", "end")):
            if event == "start":
                stack.append(elem)
                if wanted_open_at is None and local_name(elem.tag) in wanted:
                    wanted_open_at = len(stack)
                continue

            stack.pop()
            parent = stack[-1] if stack else None
            depth = len(stack) + 1

            if local_name(elem.tag) in wanted:
                if wanted_open_at == depth:
                    wanted_open_at = None
                yield elem
                _detach(parent, elem)
            elif wanted_open_at is None:
                _detach(parent, elem)
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


def parse_float(text: str | None) -> float | None:
    """A string as a finite float, or ``None``.

    ``float()`` accepts ``"NaN"``, ``"inf"`` and ``"Infinity"``, and those are
    not numbers a training file can mean — but they are numbers that reach
    ``int()`` as an uncaught ``ValueError``/``OverflowError`` several layers
    later, where the reason attached to a failed import ends up being the
    wording of a Python exception. Rejecting them at the point of entry is the
    same treatment unparseable text already gets, and keeps every downstream
    consumer able to assume its inputs are real numbers.
    """
    if text is None:
        return None
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def text_float(elem: ET.Element | None) -> float | None:
    """An element's text as a finite float, or ``None`` if it is neither."""
    if elem is None:
        return None
    return parse_float(elem.text)
