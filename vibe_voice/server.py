"""WebSocket server: PCM, WAV, or Opus in, TTS audio out."""

from __future__ import annotations

import asyncio
import hmac
import logging

from websockets.exceptions import ConnectionClosed

from vibe_voice.config import Settings
from vibe_voice.pipeline import segmenter_from_settings
from vibe_voice.session import OutEvent, VoiceSession, dumps

log = logging.getLogger("vibe_voice.server")


async def start_server(settings: Settings, engine: object, classifier: object | None = None):
    import websockets

    handler = _handler_factory(settings, engine, classifier)

    return await websockets.serve(
        handler,
        settings.ws_host,
        settings.ws_port,
        max_size=8 * 1024 * 1024,
        max_queue=256,
    )


async def serve_forever(settings: Settings, engine: object, classifier: object | None = None) -> None:
    server = await start_server(settings, engine, classifier)
    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    log.info("WebSocket 已监听 %s", sockets or f"{settings.ws_host}:{settings.ws_port}")
    await server.serve_forever()


def _handler_factory(settings: Settings, engine: object, classifier: object | None):
    async def handler(websocket) -> None:
        request = getattr(websocket, "request", None)
        path = getattr(request, "path", "") if request is not None else ""
        if not _authorized(websocket, settings.ws_token):
            log.info("拒绝未授权连接 %s", path or "/")
            await websocket.close(code=1008, reason="unauthorized")
            return
        log.info("连接 %s", path or "/")
        session = VoiceSession(
            engine,
            segmenter=segmenter_from_settings(settings),
            classifier=classifier,
            sample_rate=16000,
            max_history_messages=settings.max_history_messages,
            max_speech_ms=settings.vad_max_speech_ms,
        )
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    events = session.handle_binary(message)
                else:
                    events = session.handle_text(message)
                await _send(websocket, events)
        except ConnectionClosed:
            return
        except Exception as exc:
            log.exception("会话异常")
            try:
                await _send(websocket, session._error(f"服务器内部错误：{exc}"))
            except Exception:
                return

    return handler


async def _send(websocket, events: list[OutEvent]) -> None:
    for event in events:
        if event.binary is not None:
            await websocket.send(event.binary)
        elif event.text is not None:
            await websocket.send(dumps(event))


def _authorized(websocket, token: str) -> bool:
    if not token:
        return True
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None) or getattr(websocket, "request_headers", {}) or {}
    presented = headers.get("Authorization") or headers.get("authorization") or ""
    if isinstance(presented, (list, tuple)):
        presented = presented[0] if presented else ""
    presented = str(presented).strip()
    if presented.lower().startswith("bearer "):
        presented = presented[7:].strip()
    try:
        return hmac.compare_digest(presented, token)
    except Exception:
        return False


def run_server(settings: Settings, engine: object, classifier: object | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(serve_forever(settings, engine, classifier))
