#!/usr/bin/env python3
"""A childhood memory in one sentence, for the make.fun bounty.

Writes a single 240-character sentence to
``$MAKEFUN_OUTPUT_DIR/entry.txt`` (default ``/output``).

The sentence itself is a small, honest kitchen memory: every
evening at six, my grandmother lifted the rice-pot lid with a
folded dish towel, the steam fogged her glasses, and I stood
on a stool behind her stealing the crispy scorched rice from
the edges while she pretended not to notice. Nothing in it is
decorated: the hour, the towel, the fogged glasses and the
stolen rice are all concrete, and the ache lives in what the
sentence does not say -- that kitchen, and that grandmother,
are gone now, and the stool stands empty.

Three losing alternatives are documented in README.md: a
greeting-card sentence about missing grandmother's kitchen
(sentiment with no honest detail); a bicycle-until-dark
snapshot (concrete, but the ache is thin and the ending flat);
and a fireflies-in-a-jar summer (a familiar trope whose ending
explains its own meaning, leaving nothing unsaid).

The writer is deterministic and stdlib-only. It checks the
entry against the competition gate (UTF-8 text, 1-280
characters, one sentence on one line, no control, format,
invisible or surrogate characters), then publishes it with an
atomic rename so a reader never sees a half-written file.
"""

import os
import sys
import tempfile
import time
import unicodedata

ENTRY = (
    "At six every evening my grandmother lifted the rice-pot lid "
    "with a folded dish towel, and the steam fogged her glasses "
    "while I stood on a stool behind her, stealing the crispy "
    "scorched rice from the edges before she pretended not to "
    "notice."
)

MAX_CHARS = 280
OUTPUT_FILENAME = "entry.txt"
DEFAULT_OUTPUT_DIR = "/output"
TEMP_PREFIX = ".entry-"
TEMP_SUFFIX = ".tmp"
STALE_TEMP_MAX_AGE_SECONDS = 3600
CLEANUP_ATTEMPTS = 3
CLEANUP_RETRY_DELAY_SECONDS = 0.05


def _check_not_blank(text: str) -> None:
    """Reject entries that are empty or only whitespace."""
    if not text or not text.strip():
        raise ValueError("entry must be non-empty and not only whitespace")


def _check_length(text: str) -> None:
    """Reject entries outside the 1-280 character gate."""
    if not 1 <= len(text) <= MAX_CHARS:
        raise ValueError(
            "entry must be 1-%d characters, got %d" % (MAX_CHARS, len(text))
        )


def _check_single_line(text: str) -> None:
    """Reject entries containing a line break of either flavour.

    One sentence means one line: any newline or carriage return
    fails the gate.
    """
    if "\n" in text or "\r" in text:
        raise ValueError("entry must be a single line")


def _check_control_and_separators(text: str) -> None:
    """Reject control characters, BOM, separators and surrogates.

    Covers C0 controls and DEL, C1 controls, BOM (U+FEFF), line
    and paragraph separators (U+2028/U+2029), and lone
    surrogates (U+D800-U+DFFF). Invisible format characters are
    handled separately by :func:`_check_invisible_format`.
    """
    for ch in text:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            raise ValueError(
                "entry contains a control character U+%04X" % code
            )
        if 0x80 <= code <= 0x9F:
            raise ValueError(
                "entry contains a C1 control character U+%04X" % code
            )
        if code == 0xFEFF:
            raise ValueError("entry contains a byte-order mark U+FEFF")
        if code in (0x2028, 0x2029):
            raise ValueError(
                "entry contains a line/paragraph separator U+%04X" % code
            )
        if 0xD800 <= code <= 0xDFFF:
            raise ValueError(
                "entry contains a surrogate U+%04X" % code
            )


def _check_invisible_format(text: str) -> None:
    """Reject invisible Unicode format characters (category Cf).

    That is zero-width space, joiners, soft hyphen and the like:
    characters that take no visible space yet would survive a
    naive printable check and change the byte stream.
    """
    for ch in text:
        if unicodedata.category(ch) == "Cf":
            raise ValueError(
                "entry contains an invisible format character U+%04X"
                % ord(ch)
            )


def _run_gate_checks(text: str) -> None:
    """Run every gate check, in the order the gate states them."""
    _check_not_blank(text)
    _check_length(text)
    _check_single_line(text)
    _check_control_and_separators(text)
    _check_invisible_format(text)


def _output_dir() -> str:
    """Return the output directory.

    A set-but-blank ``MAKEFUN_OUTPUT_DIR`` (empty string, or
    whitespace only) falls back to the default instead of crashing
    ``makedirs`` on an empty path or writing into a
    whitespace-named directory.
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if raw is None:
        return DEFAULT_OUTPUT_DIR
    stripped = raw.strip()
    if not stripped:
        return DEFAULT_OUTPUT_DIR
    return stripped


def _unlink_with_retry(path: str) -> None:
    """Best-effort unlink, retried briefly against transient errors."""
    last_error = None
    for _attempt in range(CLEANUP_ATTEMPTS):
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(CLEANUP_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error


def _prune_stale_temps(out_dir: str) -> None:
    """Remove abandoned temp files from earlier interrupted runs.

    Only regular files this writer could have created (matching
    its temp prefix/suffix) and older than one hour are touched,
    so a temp file belonging to a concurrently running writer is
    never deleted, and a directory that happens to share the
    prefix can never be unlinked by mistake. Failures are
    swallowed: a stale temp file must never block the entry
    itself.
    """
    try:
        names = os.listdir(out_dir)
    except OSError:
        return
    now = time.time()
    for name in names:
        if not name.startswith(TEMP_PREFIX):
            continue
        if not name.endswith(TEMP_SUFFIX):
            continue
        path = os.path.join(out_dir, name)
        try:
            if not os.path.isfile(path):
                continue
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        if age <= STALE_TEMP_MAX_AGE_SECONDS:
            continue
        try:
            _unlink_with_retry(path)
        except OSError:
            pass


def _sync_directory(out_dir: str) -> None:
    """Flush the directory entry so the rename survives a crash."""
    dir_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _publish_atomically(out_dir: str, data: bytes) -> str:
    """Publish ``data`` as ``entry.txt`` via temp file + atomic rename.

    The staging temp file is created inside ``out_dir`` itself
    (never /tmp: the output path may be a bind-mount, and a rename
    across mounts fails with EXDEV). Returns the path of the
    published file. On any failure the temp file is removed
    (retried) and the original error propagates; the destination
    is never left half-written. A non-bytes payload is rejected
    up front so a caller bug cannot reach the file at all.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("entry payload must be bytes")
    target = os.path.join(out_dir, OUTPUT_FILENAME)
    fd, tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX
    )
    try:
        os.chmod(tmp_path, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Read-back verification before publishing.
        with open(tmp_path, "rb") as f:
            if f.read() != data:
                raise IOError("read-back verification failed")
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
        os.chmod(target, 0o644)
        _sync_directory(out_dir)
    except BaseException:
        try:
            _unlink_with_retry(tmp_path)
        except OSError:
            pass
        raise
    return target


def main() -> int:
    _run_gate_checks(ENTRY)
    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    _prune_stale_temps(out_dir)
    data = ENTRY.encode("utf-8")
    _publish_atomically(out_dir, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
