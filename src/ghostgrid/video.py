"""
Video frame extraction utilities.
"""

import base64
import contextlib
import logging

log = logging.getLogger("ghostgrid.video")


def open_video_capture(source, *, _cv2=None):
    if _cv2 is None:
        import cv2 as _cv2  # pylint: disable=import-outside-toplevel
    with contextlib.suppress(ValueError, TypeError):
        source = int(source)

    cap = _cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap, source


def extract_frames_cv2(
    source: str | int,
    fps: float = 1.0,
    max_frames: int = 0,
    *,
    _cv2=None,
) -> list[bytes]:
    if _cv2 is None:
        import cv2 as _cv2  # pylint: disable=import-outside-toplevel

    cap, _ = open_video_capture(source, _cv2=_cv2)

    video_fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))

    frames: list[bytes] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(buf.tobytes())
            if max_frames and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames (fps=%.1f, interval=%d)", len(frames), fps, frame_interval)
    return frames


def frames_to_base64(frames: list[bytes]) -> list[str]:
    """Encode raw JPEG bytes to base64 strings."""
    return [base64.b64encode(f).decode("utf-8") for f in frames]
