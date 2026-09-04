"""
app.sopal.com.au -> the drafting and execution app on Vercel.

That hostname is a CNAME to this Render service and the domain's DNS lives at a
registrar we cannot reach, so the only place the routing can be changed is
here. Requests arriving on it are handed to the Next.js deployment; every other
hostname is untouched and continues down the normal stack.

Fails open on purpose. If the upstream is unreachable, or anything in here
raises, the request falls through to the ordinary routes rather than taking the
host down with it. Set SOPAL_APP_PROXY_TARGET empty to switch it off.
"""

import os

from fastapi import Request
from fastapi.responses import StreamingResponse

APP_HOST = os.getenv("SOPAL_APP_HOST", "app.sopal.com.au").lower()
TARGET = os.getenv("SOPAL_APP_PROXY_TARGET", "https://sopal-docs.vercel.app").rstrip("/")

# Hop-by-hop headers, plus the ones the ASGI server recalculates itself.
_STRIP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "content-length", "host",
}


def install(app) -> None:
    """Attach the proxy. Called once, from server.py, right after the app."""

    @app.middleware("http")
    async def sopal_app_proxy(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":", 1)[0].lower()
        if host != APP_HOST or not TARGET:
            return await call_next(request)

        try:
            import httpx

            url = TARGET + request.url.path
            if request.url.query:
                url += "?" + request.url.query

            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in _STRIP
            }
            # So the app can tell what it is actually being served as.
            headers["x-forwarded-host"] = APP_HOST
            headers["x-forwarded-proto"] = "https"

            body = await request.body()

            client = httpx.AsyncClient(timeout=30.0, follow_redirects=False)
            upstream = await client.send(
                client.build_request(
                    request.method, url, headers=headers, content=body or None
                ),
                stream=True,
            )

            async def relay():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()

            # aiter_raw yields the bytes exactly as they arrived, so whatever
            # compression the upstream chose has to travel with them - which is
            # why content-encoding is kept while content-length is dropped.
            out = [
                (k, v) for k, v in upstream.headers.multi_items()
                if k.lower() not in _STRIP
            ]

            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                headers=dict(out),
                media_type=upstream.headers.get("content-type"),
            )
        except Exception as exc:  # noqa: BLE001 - falling through is the point
            print(f"[sopal-app-proxy] falling through: {exc}")
            return await call_next(request)
