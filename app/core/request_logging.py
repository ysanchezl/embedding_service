import logging
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging_config import request_id_var

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "X-Request-ID"
HEALTH_PATH = "/"  # polled constantly by the platform: only logged at DEBUG unless it fails


class RequestLoggingMiddleware:
    """Logs when each request arrives and how it ended, and tags it with a request id.

    The id is returned in the X-Request-ID header so a client can quote it when reporting a problem.
    Bodies and headers are never logged.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(request_id)
        label = f"{scope['method']} {scope['path']}"
        quiet = scope["path"] == HEALTH_PATH
        state = {"status": None}
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        logger.log(logging.DEBUG if quiet else logging.INFO, "%s received", label)
        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception("%s failed with an unhandled error after %.0f ms", label, elapsed)
            if state["status"] is not None:
                raise  # the response already started: nothing sensible left to send
            response = JSONResponse({"detail": "Internal server error"}, 500, {REQUEST_ID_HEADER: request_id})
            await response(scope, receive, send)
        else:
            elapsed = (time.perf_counter() - started) * 1000
            status = state["status"] or 0
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            if quiet and level == logging.INFO:
                level = logging.DEBUG
            logger.log(level, "%s -> %s in %.0f ms", label, status, elapsed)
        finally:
            request_id_var.reset(token)
