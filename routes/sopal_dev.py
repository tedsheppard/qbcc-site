"""Sopal dev chat — owner-only bridge from the live app to a Claude coding agent.

A floating chat widget in app.sopal.com.au (rendered only for the owner
account) lets Ted request changes to the app while using it. Messages are
relayed to an Anthropic Managed Agent session that has this GitHub repo
mounted in a sandboxed cloud container: the agent reads and edits the code
THERE (never on this production server), runs its checks, commits, and
pushes to main — which Render deploys.

Server-side this module is a thin relay + persistence layer:
  - POST /api/sopal-v2/dev/message   queue a user message to the session
  - GET  /api/sopal-v2/dev/stream    SSE relay of the agent's live activity
  - GET  /api/sopal-v2/dev/messages  persisted thread for the widget
  - GET  /api/sopal-v2/dev/status    config / session state
  - DELETE /api/sopal-v2/dev/messages  clear thread + start a fresh session

Requirements (Render env):
  ANTHROPIC_API_KEY        — Anthropic API key (Managed Agents beta access)
  SOPAL_DEV_GITHUB_TOKEN   — fine-grained PAT, Contents: Read & write on the repo
Optional:
  SOPAL_DEV_REPO           — default "tedsheppard/qbcc-site"
  SOPAL_DEV_MODEL          — default "claude-opus-4-8"
  SOPAL_DEV_CHAT_OWNER     — default "edwardsheppard5@gmail.com"

The agent (a persistent, versioned Anthropic resource) and its environment
are created lazily on first use and their IDs persisted in sqlite, so the
create-once / reference-by-id pattern holds across deploys.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from routes.sopal_v2 import _current_user_email, _require_persistence

router = APIRouter(prefix="/api/sopal-v2/dev", tags=["sopal-dev-chat"])

OWNER_EMAIL = os.getenv("SOPAL_DEV_CHAT_OWNER", "edwardsheppard5@gmail.com").strip().lower()
REPO_SLUG = os.getenv("SOPAL_DEV_REPO", "tedsheppard/qbcc-site").strip()
MODEL = os.getenv("SOPAL_DEV_MODEL", "claude-opus-4-8").strip()
AGENT_NAME = "Sopal Dev Agent"
ENV_NAME = "sopal-dev-env"


def _github_token() -> str:
    return (os.getenv("SOPAL_DEV_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()


def _configured() -> dict[str, bool]:
    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "github": bool(_github_token()),
    }


def _require_owner(email: str = Depends(_current_user_email)) -> str:
    if email != OWNER_EMAIL:
        raise HTTPException(status_code=403, detail="The dev chat is restricted to the owner account.")
    return email


# ---------- persistence (same sqlite file as the rest of sopal v2) ----------

def _init_tables() -> None:
    con = _require_persistence()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sopal_dev_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sopal_dev_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            role TEXT NOT NULL,           -- user | assistant | tool | system
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    con.commit()


try:  # pragma: no cover - boot-time guard mirrors routes/sopal_v2.py
    _init_tables()
except Exception as _exc:
    print(f"[sopal_dev] WARNING: dev-chat persistence offline ({_exc}).")


def _meta_get(key: str) -> str | None:
    row = _require_persistence().execute("SELECT value FROM sopal_dev_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(key: str, value: str) -> None:
    con = _require_persistence()
    con.execute(
        "INSERT INTO sopal_dev_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    con.commit()


def _meta_del(key: str) -> None:
    con = _require_persistence()
    con.execute("DELETE FROM sopal_dev_meta WHERE key = ?", (key,))
    con.commit()


def _save_message(role: str, content: str, event_id: str | None = None) -> bool:
    """Persist a thread message. Returns False if event_id was already saved."""
    if not (content or "").strip():
        return False
    con = _require_persistence()
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        con.execute(
            "INSERT INTO sopal_dev_messages (event_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (event_id, role, content, now),
        )
        con.commit()
        return True
    except Exception:
        return False  # duplicate event_id — already persisted on a prior stream


# ---------- Anthropic Managed Agents lifecycle ----------

DEV_AGENT_SYSTEM_PROMPT = """You are the live development agent for app.sopal.com.au — an Australian security-of-payment (SOPA) legal-tech app. The owner, Ted, messages you from a chat widget inside the running app and asks for changes; you implement them in the GitHub repository mounted in your workspace and deploy by pushing.

REPO: mounted at /workspace/qbcc-site (git, branch main). Pushing to main IS the production deploy — Render redeploys automatically within a few minutes.

KEY FILES
- site/assets/sopal-v2/sopal-v2.js — the whole SPA (vanilla JS IIFE, string-template render() + bind pattern; no framework, no build step).
- site/assets/sopal-v2/sopal-v2.css — its stylesheet (CSS custom properties; dark mode via [data-theme="dark"] tokens — keep both themes working).
- site/sopal-v2.html — ALWAYS bump the cache-bust ?v= query on BOTH the css and js tags whenever you change either file, or users see stale UI.
- routes/sopal_v2.py — FastAPI backend for the app. routes/sopal_dev.py is the relay you are talking through.
- server.py — main FastAPI app (be conservative here).

WORKFLOW for every request
1. git pull --rebase origin main first (the repo may have moved).
2. Read the relevant code before editing; follow the existing patterns and naming. Keep diffs minimal and focused on what was asked.
3. Verify before committing: `node --check site/assets/sopal-v2/sopal-v2.js` (install node via apt/nvm if missing) and `python3 -c "import ast; ast.parse(open('routes/sopal_v2.py').read())"` for any python you touched.
4. Bump the cache-bust version in site/sopal-v2.html when js/css changed (format ?v=YYYYMMDD-N).
5. Commit with a clear message ending with: Co-Authored-By: Sopal Dev Agent <noreply@anthropic.com>
6. git push origin main. If the push is rejected, pull --rebase and retry.

HARD RULES
- Never commit secrets, tokens, or .env content. Never edit or commit database files (*.db, *.db-shm, *.db-wal) or anything under _local_data/.
- Statutory and judicial text (BIF Act, SOP Acts, case extracts) must be reproduced VERBATIM — never paraphrase or "correct" legal text baked into prompts or UI.
- No mock/demo data: never seed fictitious projects, parties, or document text into the app.
- Australian English in all UI copy.
- If a request is ambiguous or risky (data loss, auth changes, large refactors), say so and ask Ted before proceeding rather than guessing.

REPLY STYLE: you're chatting with the owner inside his app. Be brief and concrete — what you changed, which files, that you've pushed and the deploy is rolling out (or why you stopped and what you need)."""


def _client():
    import anthropic

    return anthropic.Anthropic()


def _ensure_agent(client) -> str:
    agent_id = _meta_get("agent_id")
    if agent_id:
        return agent_id
    # Recover from lost meta (fresh disk) before creating a duplicate.
    try:
        for a in client.beta.agents.list():
            if getattr(a, "name", "") == AGENT_NAME:
                _meta_set("agent_id", a.id)
                return a.id
    except Exception:
        pass
    agent = client.beta.agents.create(
        name=AGENT_NAME,
        model=MODEL,
        system=DEV_AGENT_SYSTEM_PROMPT,
        tools=[{"type": "agent_toolset_20260401"}],
    )
    _meta_set("agent_id", agent.id)
    return agent.id


def _ensure_environment(client) -> str:
    env_id = _meta_get("environment_id")
    if env_id:
        return env_id
    try:
        for e in client.beta.environments.list():
            if getattr(e, "name", "") == ENV_NAME:
                _meta_set("environment_id", e.id)
                return e.id
    except Exception:
        pass
    env = client.beta.environments.create(
        name=ENV_NAME,
        config={"type": "cloud", "networking": {"type": "unrestricted"}},
    )
    _meta_set("environment_id", env.id)
    return env.id


def _session_alive(client, session_id: str) -> bool:
    try:
        s = client.beta.sessions.retrieve(session_id)
        return getattr(s, "status", "terminated") != "terminated" and not getattr(s, "archived_at", None)
    except Exception:
        return False


def _ensure_session(client) -> str:
    session_id = _meta_get("session_id")
    if session_id and _session_alive(client, session_id):
        return session_id
    agent_id = _ensure_agent(client)
    env_id = _ensure_environment(client)
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title="Sopal dev chat",
        resources=[{
            "type": "github_repository",
            "url": f"https://github.com/{REPO_SLUG}",
            "authorization_token": _github_token(),
            "mount_path": "/workspace/qbcc-site",
            "checkout": {"type": "branch", "name": "main"},
        }],
    )
    _meta_set("session_id", session.id)
    return session.id


# ---------- endpoints ----------

def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/status")
def dev_status(email: str = Depends(_require_owner)) -> dict[str, Any]:
    conf = _configured()
    session_id = _meta_get("session_id")
    return {
        "configured": conf["anthropic"] and conf["github"],
        "missing": [k for k, ok in (("ANTHROPIC_API_KEY", conf["anthropic"]), ("SOPAL_DEV_GITHUB_TOKEN", conf["github"])) if not ok],
        "repo": REPO_SLUG,
        "model": MODEL,
        "hasSession": bool(session_id),
    }


@router.get("/messages")
def dev_messages(email: str = Depends(_require_owner)) -> dict[str, Any]:
    rows = _require_persistence().execute(
        "SELECT role, content, created_at FROM sopal_dev_messages ORDER BY id ASC LIMIT 500"
    ).fetchall()
    return {"messages": [{"role": r["role"], "content": r["content"], "at": r["created_at"]} for r in rows]}


@router.delete("/messages")
def dev_clear(email: str = Depends(_require_owner)) -> dict[str, Any]:
    con = _require_persistence()
    con.execute("DELETE FROM sopal_dev_messages")
    con.commit()
    # Archive the old session so the next message starts a fresh container
    # (a fresh clone of main, no stale workspace state).
    session_id = _meta_get("session_id")
    if session_id:
        try:
            _client().beta.sessions.archive(session_id)
        except Exception:
            pass
        _meta_del("session_id")
    return {"cleared": True}


@router.post("/message")
async def dev_message(payload: dict[str, Any], email: str = Depends(_require_owner)) -> dict[str, Any]:
    message = str(payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")
    if len(message) > 20_000:
        raise HTTPException(status_code=400, detail="Message is too long.")
    conf = _configured()
    if not (conf["anthropic"] and conf["github"]):
        raise HTTPException(status_code=503, detail="Dev chat is not configured on the server (ANTHROPIC_API_KEY / SOPAL_DEV_GITHUB_TOKEN).")

    client = _client()
    try:
        session_id = await asyncio.to_thread(_ensure_session, client)
        await asyncio.to_thread(
            client.beta.sessions.events.send,
            session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": message}]}],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the dev agent: {exc}") from exc

    _save_message("user", message)
    return {"ok": True, "sessionId": session_id}


def _summarise_tool_use(ev: Any) -> str:
    """One status line per tool call, e.g. '$ git status' or 'edit sopal-v2.js'."""
    name = getattr(ev, "name", "") or getattr(ev, "tool_name", "") or "tool"
    inp = getattr(ev, "input", None) or {}
    if not isinstance(inp, dict):
        inp = {}
    if name == "bash":
        cmd = str(inp.get("command") or "").strip().splitlines()[0][:120]
        return f"$ {cmd}" if cmd else "$ (bash)"
    path = str(inp.get("path") or inp.get("file_path") or inp.get("pattern") or "").strip()
    short = path.rsplit("/", 1)[-1][:80] if path else ""
    return f"{name} {short}".strip()


def _event_text(ev: Any) -> str:
    parts = []
    for block in getattr(ev, "content", None) or []:
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if p)


@router.get("/stream")
async def dev_stream(email: str = Depends(_require_owner)) -> StreamingResponse:
    conf = _configured()
    if not (conf["anthropic"] and conf["github"]):
        raise HTTPException(status_code=503, detail="Dev chat is not configured on the server.")
    session_id = _meta_get("session_id")
    if not session_id:
        raise HTTPException(status_code=404, detail="No active dev session — send a message first.")

    client = _client()
    loop_holder: dict[str, Any] = {}
    queue: asyncio.Queue = asyncio.Queue()
    stop_flag = threading.Event()

    def put(item: dict[str, Any]) -> None:
        loop = loop_holder.get("loop")
        if loop:
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def pump() -> None:
        """Open the live stream first, replay history, then tail live events.

        The documented consolidation pattern: SSE has no replay, so on every
        connect we overlap the stream with an events.list() and dedupe by
        event id (persistence also dedupes via the UNIQUE event_id column).
        """
        seen: set[str] = set()

        def relay(ev: Any, live: bool) -> None:
            ev_id = getattr(ev, "id", "") or ""
            ev_type = getattr(ev, "type", "") or ""
            if ev_id and ev_id in seen:
                # Already relayed in this connection — but terminal checks
                # below still ran the first time we saw it.
                return
            if ev_id:
                seen.add(ev_id)
            if ev_type == "agent.message":
                text = _event_text(ev)
                if text:
                    fresh = _save_message("assistant", text, event_id=ev_id or None)
                    if fresh or live:
                        put({"kind": "assistant", "text": text})
            elif ev_type in ("agent.tool_use", "agent.mcp_tool_use"):
                if live:
                    put({"kind": "tool", "text": _summarise_tool_use(ev)})
            elif ev_type == "agent.thinking":
                pass  # opaque; the tool lines carry the progress story
            elif ev_type == "session.error":
                err = getattr(ev, "error", None)
                msg = getattr(err, "message", None) or str(err or "Session error")
                put({"kind": "error", "text": str(msg)[:500]})
            elif ev_type == "session.status_terminated":
                put({"kind": "done", "text": "Session terminated — the next message starts a fresh one."})
                _meta_del("session_id")
                stop_flag.set()
            elif ev_type == "session.status_idle":
                stop = getattr(ev, "stop_reason", None)
                stop_type = getattr(stop, "type", "") if stop else ""
                if live and stop_type != "requires_action":
                    put({"kind": "done", "text": ""})
                    stop_flag.set()
            elif ev_type == "session.status_running":
                if live:
                    put({"kind": "status", "text": "Working…"})

        try:
            with client.beta.sessions.events.stream(session_id) as stream:
                # History pass (also persists anything missed while the
                # browser / this server was away).
                try:
                    for ev in client.beta.sessions.events.list(session_id):
                        relay(ev, live=False)
                except Exception:
                    pass
                # If the session is already idle and the last user message
                # is not fresh (i.e. nothing should be in flight), end the
                # turn rather than holding the connection silently. A freshly
                # queued message flips the session to running within moments,
                # so skip the early "done" in that window.
                try:
                    s = client.beta.sessions.retrieve(session_id)
                    if getattr(s, "status", "") == "idle":
                        row = _require_persistence().execute(
                            "SELECT created_at FROM sopal_dev_messages WHERE role = 'user' ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                        fresh = False
                        if row:
                            try:
                                sent = datetime.fromisoformat(row["created_at"].rstrip("Z"))
                                fresh = (datetime.utcnow() - sent).total_seconds() < 90
                            except Exception:
                                fresh = False
                        if not fresh:
                            put({"kind": "done", "text": ""})
                            stop_flag.set()
                except Exception:
                    pass
                for ev in stream:
                    relay(ev, live=True)
                    if stop_flag.is_set():
                        break
        except Exception as exc:
            put({"kind": "error", "text": f"Stream dropped: {exc}"})
        finally:
            put({"kind": "eos", "text": ""})

    async def event_stream():
        loop_holder["loop"] = asyncio.get_running_loop()
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        yield _sse("status", {"message": "Connected to the dev agent."})
        idle_done_sent = False
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=600)
                except asyncio.TimeoutError:
                    yield _sse("status", {"message": "Still connected…"})
                    continue
                kind = item.get("kind")
                if kind == "eos":
                    break
                if kind == "assistant":
                    yield _sse("assistant", {"text": item.get("text", "")})
                elif kind == "tool":
                    yield _sse("tool", {"text": item.get("text", "")})
                elif kind == "status":
                    yield _sse("status", {"message": item.get("text", "")})
                elif kind == "error":
                    yield _sse("error", {"message": item.get("text", "")})
                elif kind == "done":
                    if not idle_done_sent:
                        idle_done_sent = True
                        yield _sse("done", {"message": item.get("text", "")})
                    if stop_flag.is_set():
                        break
        finally:
            stop_flag.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
