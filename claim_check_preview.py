"""Local dev preview server for /claim-check ONLY.

This is a devtime-only harness. It does NOT import server.py, does NOT
touch the production database, does NOT replicate site-wide functionality.
It exists so the /claim-check page can be previewed on localhost without
needing the /var/data/ Render disk or the full production dependency set.

Run:
    uvicorn claim_check_preview:app --reload --port 8000

Then open http://localhost:8000/claim-check
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from routes.claim_check import router as claim_check_router, redirect_router as claim_check_redirect
from routes.contract_assist import router as contract_assist_router
from services.contract_assist.bif_act_index import register_startup as ca_bif_register_startup

ROOT = Path(__file__).parent.resolve()
SITE_DIR = ROOT / "site"

app = FastAPI(title="Sopal claim-check preview")
app.include_router(claim_check_router)
app.include_router(claim_check_redirect)
app.include_router(contract_assist_router)
ca_bif_register_startup(app)


@app.get("/")
async def root() -> dict:
    return {
        "preview": True,
        "page": "http://localhost:8000/claim-check",
        "api_health": "http://localhost:8000/api/claim-check/health",
    }


@app.get("/{path_name:path}")
async def serve_static(path_name: str):
    # Mirror server.py's catch-all: /claim-check -> site/claim-check.html,
    # /assets/foo.png -> site/assets/foo.png, etc.
    safe_root = SITE_DIR.resolve()

    candidates = [
        SITE_DIR / f"{path_name}.html",
        SITE_DIR / path_name,
        SITE_DIR / path_name.rstrip("/") / "index.html",
    ]
    for p in candidates:
        try:
            resolved = p.resolve()
        except (OSError, ValueError):
            continue
        # Prevent path traversal.
        if not str(resolved).startswith(str(safe_root)):
            continue
        if resolved.is_file():
            return FileResponse(str(resolved))

    raise HTTPException(status_code=404, detail=f"Not found: /{path_name}")
