"""Compatibility adapter for non-terminal Marketplace progress events."""

from typing import Any

try:
    from shared.services.events import emitter
    from shared.services.events.types import EventType
except ModuleNotFoundError:  # pragma: no cover - legacy local SDK only
    emitter = None
    EventType = None


def emit_progress(message: str, stage: str, **metadata: Any) -> None:
    """Emit progress when supported; never own terminal success/failure events."""
    if emitter is None or EventType is None:
        return
    emitter().emit_event(
        event_type=EventType.PROGRESS_UPDATE,
        message=message,
        metadata={"stage": stage, **metadata},
    )
