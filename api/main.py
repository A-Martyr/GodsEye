"""GodsEye API — city-wide ANPR platform.

    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from api.routes import router
from api.state import state
from core import reach


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.bootstrap()
    # Settle the token before anything can ask for it, so the API and the
    # dashboard cannot race each other into generating two different ones.
    reach.token()
    # Load the reader now, off the request path. The first read of a frame
    # costs 3.3 s against a 64 ms median, and left alone that lands on whoever
    # calls /api/read first - or on the simulator's first tick, stalling the
    # live feed at exactly the moment someone starts watching it.
    if config.WARM_READER_ON_START:
        from anpr.ocr import warm_async
        warm_async()
    state.sim_task = asyncio.create_task(state.sim_loop())
    yield
    state.sim_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await state.sim_task


app = FastAPI(
    title="GodsEye",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "City-wide ANPR platform: a high-accuracy plate reader, spatial-temporal "
        f"trajectory reconstruction and macro traffic analytics over the {config.CITY_NAME} "
        "camera network."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Everything this port serves is what a tunnel publishes - a tunnel maps an
# origin, not a path. Without this, sending someone a link to the judge
# console would also hand them /docs, every sighting in the database and the
# whole ANPR surface. CORS does not help: it is already open to "*", and it
# constrains browsers rather than the person holding curl.
#
# The Host header separates the two audiences. A phone in the room asks for
# 192.168.x.x; a browser arriving down the tunnel asks for
# something.trycloudflare.com. The client address cannot be used instead,
# because cloudflared connects to us over loopback - every tunnelled request
# would look local.
PUBLIC_PATHS = ("/j", "/judge", "/api/judge")


@app.middleware("http")
async def restrict_public_origin(request: Request, call_next):
    path = request.url.path
    if reach.is_public_host(request.headers.get("host", "")) and not (
            path in PUBLIC_PATHS or any(path.startswith(p + "/") for p in PUBLIC_PATHS)):
        # A hosted public demo (config.PUBLIC_DEMO) opens the reads to the
        # public hostname and nothing else: every write is a GET-less method,
        # so refusing those refuses every change a stranger could make. The
        # judge paths above keep their own token guard either way.
        if not config.PUBLIC_DEMO:
            return JSONResponse({"detail": "not found"}, status_code=404)
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": "This is the read-only public demo. "
                                           "Run GodsEye locally to change anything."},
                                status_code=403)
    return await call_next(request)


app.include_router(router)

# The control room: plain HTML, CSS and JS in web/, served by this same app so
# the operator console and the API share one origin and one process.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(WEB_DIR / "index.html")


# The page files a browser asks for at the root, and the privacy page. The
# control room's code, styles and fonts are about 330 KB as written; compressed
# they cross the network at a fraction of that.
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/privacy", include_in_schema=False)
def privacy_page():
    return FileResponse(WEB_DIR / "privacy.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=301)


@app.get("/robots.txt", include_in_schema=False)
def robots():
    # The control room and the privacy page may be indexed; the API may not.
    return PlainTextResponse("User-agent: *\nAllow: /$\nAllow: /privacy\nAllow: /static/\nDisallow: /api/\nDisallow: /j\nDisallow: /docs\n")


NOT_FOUND_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Not found · GodsEye</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml"><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0c1015;color:#a7b2bf;
font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;text-align:center}
h1{color:#e7ecf1;font-size:24px;margin:0 0 8px}a{color:#4d84ff}</style></head><body><main>
<h1>Nothing at this address</h1><p>The camera network is still watching, just not here.</p>
<p><a href="/">Open the control room</a></p></main></body></html>"""


@app.exception_handler(StarletteHTTPException)
async def not_found_for_people(request: Request, exc: StarletteHTTPException):
    # A person who mistypes an address gets a page with a way back; the API,
    # and anything that did not ask for HTML, keeps its JSON errors.
    if (exc.status_code == 404 and not request.url.path.startswith("/api")
            and "text/html" in request.headers.get("accept", "")):
        return HTMLResponse(NOT_FOUND_PAGE, status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))


@app.get("/api", include_in_schema=False)
def service_info():
    return {"service": "GodsEye", "docs": "/docs", "health": "/api/health",
            "judge_console": "/judge", "control_room": "/"}


@app.get("/j", include_in_schema=False)
def judge_short(request: Request):
    """The shortest path that still means something.

    This is the URL that gets read out, typed into a phone, or pasted into a
    message at the start of a demonstration, so every character removed is one
    fewer chance to mistype it. /judge stays as the readable alias.
    """
    from api.routes import judge_console

    return judge_console(request)


@app.get("/j/{token}", include_in_schema=False)
def judge_short_token(request: Request, token: str):
    """The same console, with the token carried in the path.

    This is the form that gets sent to someone who is not in the room. Putting
    the token in the link rather than asking for it at a prompt keeps the
    original promise - whoever holds it presses a button and types nothing -
    while leaving the public hostname useless to anyone who merely stumbles
    onto it.
    """
    from api.routes import judge_console

    return judge_console(request)


@app.get("/judge", include_in_schema=False)
def judge(request: Request):
    """The judge console, at a path short enough to read out loud.

    Mounted on the app rather than the router so it is /judge and not
    /api/judge: someone is going to type this into a phone while standing in
    front of the stand, and every character is one more chance to mistype.
    """
    from api.routes import judge_console

    return judge_console(request)
