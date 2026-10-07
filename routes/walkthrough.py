"""15-minute Workspace walkthroughs, booked from sopal.com.au/workspace.

The times offered are a fixed list (Brisbane time), not a calendar feed: a
handful of slots on each working day and some Saturday mornings. A booked
slot is no longer offered. Each booking emails Sopal a calendar invite and
confirms to the person who booked.
"""
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import aiosmtplib
from fastapi import APIRouter, Body, HTTPException

router = APIRouter()

BRISBANE = timezone(timedelta(hours=10))  # no daylight saving
MINUTES = 15
TO = os.getenv("WALKTHROUGH_TO", "info@sopal.com.au")

SLOTS = {
    "2026-10-09": ["07:50", "13:05"],
    "2026-10-10": ["10:25", "10:45"],
    "2026-10-12": ["08:00", "17:50"],
    "2026-10-13": ["13:15"],
    "2026-10-14": ["08:00", "18:10"],
    "2026-10-15": ["12:45", "18:15"],
    "2026-10-16": ["13:25", "17:55"],
    "2026-10-17": ["09:10", "09:30", "10:15"],
    "2026-10-19": ["07:45", "13:20", "17:35"],
    "2026-10-20": ["08:00", "12:45"],
    "2026-10-21": ["17:55"],
    "2026-10-22": ["13:00", "17:35"],
    "2026-10-23": ["08:15", "12:15", "13:15"],
    "2026-10-24": ["09:15", "09:45", "10:45"],
    "2026-10-26": ["07:50", "08:15", "17:55"],
    "2026-10-27": ["13:00"],
    "2026-10-28": ["12:45"],
    "2026-10-29": ["12:50"],
    "2026-10-30": ["07:45", "17:40"],
    "2026-10-31": ["09:25", "09:50", "10:30"],
}

_path = "/var/data/walkthroughs.db" if os.path.isdir("/var/data") else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_local_data", "walkthroughs.db")
os.makedirs(os.path.dirname(_path), exist_ok=True)
_con = sqlite3.connect(_path, check_same_thread=False)
_con.execute("""CREATE TABLE IF NOT EXISTS walkthroughs (
  slot TEXT PRIMARY KEY, name TEXT, email TEXT, firm TEXT, note TEXT, created_at TEXT)""")
_con.commit()
_lock = threading.Lock()


def _start(slot: str) -> datetime:
    return datetime.strptime(slot, "%Y-%m-%dT%H:%M").replace(tzinfo=BRISBANE)


def _open() -> list[str]:
    booked = {r[0] for r in _con.execute("SELECT slot FROM walkthroughs")}
    soon = datetime.now(BRISBANE) + timedelta(hours=12)
    return [f"{d}T{t}" for d, ts in SLOTS.items() for t in ts
            if f"{d}T{t}" not in booked and _start(f"{d}T{t}") > soon]


@router.get("/api/walkthrough/slots")
def slots():
    return {"timezone": "Brisbane time (AEST)", "minutes": MINUTES, "slots": _open()}


def _ics(uid: str, start: datetime, name: str, email: str, firm: str) -> str:
    fmt = lambda t: t.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    who = f"{name}, {firm}" if firm else name
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Sopal//Walkthrough//EN", "METHOD:REQUEST",
        "BEGIN:VEVENT", f"UID:{uid}@sopal.com.au", f"DTSTAMP:{fmt(datetime.now(timezone.utc))}",
        f"DTSTART:{fmt(start)}", f"DTEND:{fmt(start + timedelta(minutes=MINUTES))}",
        f"SUMMARY:Sopal Workspace walkthrough ({who})",
        f"ORGANIZER;CN=Sopal:mailto:{TO}",
        f"ATTENDEE;CN={name};RSVP=TRUE:mailto:{email}",
        "END:VEVENT", "END:VCALENDAR", ""])


async def _send(to: str, subject: str, body: str, ics: str | None = None, reply_to: str | None = None):
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM_EMAIL", "info@sopal.com.au")
    msg["To"] = to
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    if ics:
        msg.add_attachment(ics.encode(), maintype="text", subtype="calendar", filename="walkthrough.ics")
    await aiosmtplib.send(
        msg, hostname=os.getenv("SMTP_HOST", "smtp.office365.com"), port=int(os.getenv("SMTP_PORT", "587")),
        start_tls=True, username=os.getenv("SMTP_USERNAME", "info@sopal.com.au"), password=os.getenv("SMTP_PASSWORD"))


@router.post("/api/walkthrough/book")
async def book(b: dict = Body(...)):
    slot = str(b.get("slot", ""))
    name = str(b.get("name", "")).strip()[:120]
    email = str(b.get("email", "")).strip()[:200]
    firm = str(b.get("firm", "")).strip()[:160]
    note = str(b.get("note", "")).strip()[:1000]
    if not name or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Enter your name and email.")
    with _lock:
        if slot not in _open():
            raise HTTPException(409, "That time has just been taken. Choose another.")
        _con.execute("INSERT INTO walkthroughs VALUES (?,?,?,?,?,?)",
                     (slot, name, email, firm, note, datetime.now(timezone.utc).isoformat()))
        _con.commit()
    start = _start(slot)
    when = start.strftime("%A %-d %B at %-I:%M%p").replace("AM", "am").replace("PM", "pm") + " (Brisbane time)"
    ics = _ics(uuid.uuid4().hex, start, name, email, firm)
    try:
        await _send(TO, f"Walkthrough booked: {name}{', ' + firm if firm else ''}, {when}",
                    f"{name} <{email}>\nFirm: {firm or '-'}\nWhen: {when}\n\n{note or ''}\n\nSend them a Teams link for the call.",
                    ics, reply_to=email)
        await _send(email, "Your Sopal Workspace walkthrough",
                    f"Hi {name.split()[0]},\n\nThanks for booking a 15-minute walkthrough of Sopal Workspace on {when}.\n\n"
                    "We'll send you a Microsoft Teams link before the call. If you need to change the time, just reply to this email.\n\n"
                    "Kind regards\nSopal", ics)
    except Exception as e:  # the booking stands; Sopal can follow up from the database
        print(f"Walkthrough email failed for {slot}: {e}")
    return {"ok": True, "when": when}


# ── Registrations of interest in Sopal Workspace (sopal.com.au/workspace/register) ──

_con.execute("""CREATE TABLE IF NOT EXISTS workspace_interest (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT, email TEXT, firm TEXT, phone TEXT, role TEXT,
  users TEXT, states TEXT, acts TEXT, library TEXT, notes TEXT, walkthrough INTEGER)""")
_con.commit()


@router.post("/api/workspace/interest")
async def register_interest(b: dict = Body(...)):
    s = lambda k, n=200: str(b.get(k) or "").strip()[:n]
    name, email, firm = s("name", 120), s("email"), s("firm", 160)
    if not name or not firm or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Please give your name, firm and work email.")
    states = ", ".join(str(x)[:4] for x in (b.get("states") or [])[:8] if isinstance(x, str))
    row = dict(name=name, email=email, firm=firm, phone=s("phone", 40), role=s("role", 60), users=s("users", 30), states=states,
               acts=s("acts", 30), library=s("library", 40), notes=s("notes", 2000), walkthrough=1 if b.get("walkthrough") else 0)
    with _lock:
        _con.execute("INSERT INTO workspace_interest (created_at, name, email, firm, phone, role, users, states, acts, library, notes, walkthrough) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), *row.values()))
        _con.commit()
    lines = "\n".join(f"{k.capitalize()}: {v}" for k, v in row.items() if v not in ("", 0))
    try:
        await _send(TO, f"Workspace interest: {name}, {firm}{' (wants a walkthrough)' if row['walkthrough'] else ''}", lines, reply_to=email)
        await _send(email, "Sopal Workspace: we've got your registration",
                    f"Hi {name.split()[0]},\n\nThanks for registering {firm}'s interest in Sopal Workspace. We'll be in touch about access soon"
                    f"{' and to arrange a walkthrough' if row['walkthrough'] else ''}.\n\nKind regards\nSopal")
    except Exception as e:  # the registration stands; Sopal can follow up from the database
        print(f"Workspace interest email failed for {email}: {e}")
    return {"ok": True}
