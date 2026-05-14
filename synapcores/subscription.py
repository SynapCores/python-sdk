"""
Real-time subscription support for SynapCores Python SDK.

v0.2.0: WebSocket auth uses the ticket exchange flow.
  1. POST /v1/ws/ticket -> {token, expiresAt}
  2. Open `ws://host:port/ws?token={ticket}`  (root /ws, not /v1/ws)
"""

import asyncio
import json
from urllib.parse import quote
import websockets
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
from datetime import datetime
import logging

from .models import SubscriptionEvent, Document

if TYPE_CHECKING:
    from .collection import Collection

logger = logging.getLogger(__name__)


class Subscription:
    """Manages real-time subscriptions to collection changes."""

    def __init__(
        self,
        collection: "Collection",
        filter: Optional[Dict[str, Any]] = None,
        on_change: Optional[Callable[[SubscriptionEvent], None]] = None,
    ):
        self.collection = collection
        self.filter = filter or {}
        self.on_change = on_change
        self._websocket = None
        self._task = None
        self._running = False

    def _build_ws_url(self) -> str:
        client = self.collection.client
        ws_base = client._ws_base_url() if hasattr(client, "_ws_base_url") else (
            ("wss" if client.use_https else "ws")
            + f"://{client.host}:{client.port}"
        )
        ticket = client.create_ws_ticket()
        token = ticket.get("token") or ticket.get("ticket") or ""
        return f"{ws_base}/ws?token={quote(token)}"

    async def connect(self) -> None:
        """Connect to WebSocket and start listening."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for WebSocket messages."""
        try:
            url = self._build_ws_url()
            async with websockets.connect(url) as websocket:
                self._websocket = websocket

                # Subscribe to collection
                await self._subscribe()

                # Listen for messages
                while self._running:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=30.0,
                        )
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error(f"Error in subscription: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._running = False
        finally:
            self._websocket = None

    async def _subscribe(self) -> None:
        """Send subscription request."""
        subscribe_msg = {
            "type": "subscribe",
            "collection": self.collection.name,
            "filter": self.filter,
        }
        await self._websocket.send(json.dumps(subscribe_msg))

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            if data.get("type") == "error":
                logger.error(f"Subscription error: {data.get('message')}")
                return

            if data.get("type") == "change":
                event = SubscriptionEvent(
                    operation=data["operation"],
                    collection=data["collection"],
                    document=Document(**data["document"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    sequence=data["sequence"],
                )

                if self.on_change:
                    # Call handler in thread pool to avoid blocking
                    asyncio.create_task(
                        asyncio.to_thread(self.on_change, event)
                    )

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def close(self) -> None:
        """Close the subscription."""
        self._running = False

        if self._websocket:
            await self._websocket.close()

        if self._task:
            await self._task

    def __aiter__(self):
        """Async iterator support."""
        return self

    async def __anext__(self) -> SubscriptionEvent:
        """Get next event from subscription."""
        if not self._running:
            raise StopAsyncIteration

        # This would need a queue implementation for proper async iteration
        raise NotImplementedError("Async iteration not yet implemented")
