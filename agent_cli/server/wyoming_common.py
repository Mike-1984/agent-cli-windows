"""Shared base class for Wyoming protocol event handlers."""

from __future__ import annotations

import logging

from wyoming.server import AsyncEventHandler

logger = logging.getLogger(__name__)


class ResilientEventHandler(AsyncEventHandler):
    """Wyoming event handler that tolerates abrupt client disconnects.

    On Windows, a client that vanishes mid-connection (process killed,
    network drop) surfaces from the event read loop as a raw
    ConnectionResetError instead of a clean EOF. Left uncaught, that
    exception escapes the handler's task, which nothing awaits, so asyncio
    logs it as "Task exception was never retrieved" with a scary traceback
    even though the server itself is unaffected.
    """

    async def run(self) -> None:
        """Run the event loop, swallowing abrupt-disconnect errors."""
        try:
            await super().run()
        except ConnectionError as exc:
            logger.debug("Wyoming client disconnected abruptly: %s", exc)
