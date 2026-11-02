"""Upload validation: size limits and format sniffing by magic bytes.

The client-supplied ``Content-Type`` and file extension are not trusted; the
first bytes of the payload decide whether it is a supported audio container.
"""

from __future__ import annotations

from collections.abc import Callable

from speech_grammar.domain.errors import (
    AudioTooLargeError,
    InvalidAudioError,
    UnsupportedMediaError,
)

SUPPORTED_FORMATS = ("wav", "mp3", "flac", "ogg", "webm", "mp4", "aac", "aiff")


def _is_mp3(b: bytes) -> bool:
    if b[:3] == b"ID3":
        return True
    # MPEG audio frame sync: 11 set bits, layer bits != 00 (00 would be ADTS AAC).
    return len(b) > 1 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0 and (b[1] & 0x06) != 0


def _is_adts_aac(b: bytes) -> bool:
    return len(b) > 1 and b[0] == 0xFF and (b[1] & 0xF6) == 0xF0


_SIGNATURES: tuple[tuple[str, Callable[[bytes], bool]], ...] = (
    ("wav", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WAVE"),
    ("flac", lambda b: b[:4] == b"fLaC"),
    ("ogg", lambda b: b[:4] == b"OggS"),
    ("webm", lambda b: b[:4] == b"\x1a\x45\xdf\xa3"),
    ("mp4", lambda b: b[4:8] == b"ftyp"),
    ("aiff", lambda b: b[:4] == b"FORM" and b[8:12] in (b"AIFF", b"AIFC")),
    ("aac", _is_adts_aac),
    ("mp3", _is_mp3),
)


def sniff_format(data: bytes) -> str | None:
    """Detect the audio container from its leading bytes.

    Args:
        data: Raw uploaded bytes.

    Returns:
        A format name from :data:`SUPPORTED_FORMATS` or ``None`` when unknown.
    """
    head = data[:16]
    for name, matches in _SIGNATURES:
        if matches(head):
            return name
    return None


def validate_upload(data: bytes, *, max_bytes: int) -> str:
    """Validate an uploaded audio payload before decoding.

    Args:
        data: Raw uploaded bytes.
        max_bytes: Maximum accepted size.

    Returns:
        The sniffed format name.

    Raises:
        InvalidAudioError: The file is empty.
        AudioTooLargeError: The file is larger than ``max_bytes``.
        UnsupportedMediaError: The bytes are not a supported audio container.
    """
    if not data:
        raise InvalidAudioError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise AudioTooLargeError(f"Audio file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    fmt = sniff_format(data)
    if fmt is None:
        raise UnsupportedMediaError(
            "Unsupported audio format. Use one of: " + ", ".join(SUPPORTED_FORMATS) + "."
        )
    return fmt
