from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cloud_server import app

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/app", include_in_schema=False)
def pwa_app():
    return FileResponse(
        WEB_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/manifest.webmanifest", include_in_schema=False)
def pwa_manifest():
    return FileResponse(
        WEB_DIR / "manifest.webmanifest",
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/sw.js", include_in_schema=False)
def pwa_service_worker():
    return FileResponse(
        WEB_DIR / "sw.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",
            "Service-Worker-Allowed": "/",
        },
    )
