"""
mp4_probe.py — peeks at an MP4's top-level box structure to determine
whether it can be read while still growing.

MP4 files are organized as a sequence of top-level "boxes" (ftyp, moov,
mdat, free, ...), each starting with an 8-byte header: a 4-byte size and a
4-byte type code. The `moov` box holds the index needed to make sense of
`mdat` (the actual media data) -- most recording/muxing tools write `moov`
LAST, once they finally know the file's total duration/size, which means
the file is structurally unreadable as a stream until it's completely
finished. Fragmented MP4 and "fast-start" MP4 write `moov` first instead,
which makes progressive/live reading possible.

This is a read-only probe -- it never modifies the file, and stops as
soon as it finds a definitive answer.
"""
import struct

STREAMABLE = "streamable"
NOT_STREAMABLE = "not_streamable"
UNKNOWN = "unknown"


def probe_mp4_streamability(file_path, max_boxes=64):
    """Returns STREAMABLE if `moov` appears before `mdat` in the file's
    top-level box order, NOT_STREAMABLE if `mdat` appears first, or
    UNKNOWN if neither could be determined yet (not enough bytes written
    so far, corrupt/non-MP4 data, or an unusually deep box structure)."""
    try:
        with open(file_path, "rb") as f:
            pos = 0
            for _ in range(max_boxes):
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    return UNKNOWN  # nicht genug Daten (noch) geschrieben, um zu entscheiden
                size, box_type_raw = struct.unpack(">I4s", header)
                box_type = box_type_raw.decode("ascii", errors="replace")

                if box_type == "moov":
                    return STREAMABLE
                if box_type == "mdat":
                    return NOT_STREAMABLE

                if size == 1:
                    # 64-bit "largesize" -- die echte Boxgröße steht in den
                    # nächsten 8 Bytes, zählt die 16 Header-Bytes selbst mit.
                    largesize_bytes = f.read(8)
                    if len(largesize_bytes) < 8:
                        return UNKNOWN
                    size = struct.unpack(">Q", largesize_bytes)[0]
                elif size == 0:
                    # Box erstreckt sich bis zum Dateiende -- kann nicht
                    # sinnvoll übersprungen werden, um zur nächsten zu
                    # gelangen. Sollte bei einer wachsenden Datei ohnehin
                    # nicht vorkommen (das wäre schon die letzte Box).
                    return UNKNOWN
                elif size < 8:
                    return UNKNOWN  # ungültige/korrupte Boxgröße

                pos += size
        return UNKNOWN  # max_boxes erreicht, ohne moov/mdat gefunden -- ungewöhnlich, lieber ehrlich "unklar" als raten
    except Exception:
        return UNKNOWN
