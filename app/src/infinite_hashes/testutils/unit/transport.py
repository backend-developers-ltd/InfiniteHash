import asyncio
import collections

import turbobt.substrate.transports.base
from turbobt.substrate._models import Request, Response


class MockedTransport(turbobt.substrate.transports.base.BaseTransport):
    def __init__(self):
        self.responses = {}
        self.subscriptions = collections.defaultdict(asyncio.Queue)
        # Capture all requests for assertions in tests
        self.requests: list[Request] = []
        # Keep last submitted extrinsic hex for convenience
        self.last_extrinsic_hex: str | None = None

    async def send(self, request: Request) -> Response:
        # Record every request
        self.requests.append(request)

        # Built-in helpers for common author_* flows so higher-level code can progress
        if request.method == "author_submitExtrinsic":
            # Record submitted extrinsic hex payload
            try:
                self.last_extrinsic_hex = request.params[0]
            except Exception:
                self.last_extrinsic_hex = None
            return Response(request=request, result="0x" + ("ab" * 32), error=None)

        if request.method == "author_submitAndWatchExtrinsic":
            # Create a synthetic subscription and send a finalized event ASAP
            raw_id = b"\x01"
            sub_id = f"0x{raw_id.hex()}"
            # Record submitted extrinsic hex payload if present
            try:
                if isinstance(request.params, dict):
                    self.last_extrinsic_hex = request.params.get("bytes")
                else:
                    self.last_extrinsic_hex = request.params[0]
            except Exception:
                self.last_extrinsic_hex = None
            q = self.subscribe(sub_id)

            async def _finalize():
                await asyncio.sleep(0)
                await q.put({"finalized": "0x" + ("cd" * 32)})

            asyncio.create_task(_finalize())
            return Response(request=request, result=raw_id, error=None)

        if request.method in ("author_unwatchExtrinsic",):
            return Response(request=request, result=True, error=None)

        # State call handling (supports per-method override dict)
        if request.method in self.responses:
            response = self.responses[request.method]

            if request.method == "state_call":
                try:
                    response = response[request.params["name"]]
                except KeyError:
                    response = {}
            # Allow dynamic response functions
            if callable(response):
                try:
                    response = response(request)
                except Exception:
                    response = {"error": {"message": "dynamic handler error"}}
            return Response(
                request=request,
                result=response.get("result"),
                error=response.get("error"),
            )

        # Default fall-through: return "no error" with null result
        return Response(request=request, result=None, error=None)

    def subscribe(self, subscription_id) -> asyncio.Queue:
        return self.subscriptions[subscription_id]

    def unsubscribe(self, subscription_id) -> asyncio.Queue | None:
        return self.subscriptions.pop(subscription_id, None)
