"""Web UI: a small chat page served by FastAPI.

    uvicorn noteflow.app:app --reload
    open http://127.0.0.1:8000
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from noteflow import db, embeddings
from noteflow.orchestrator import Agent

load_dotenv()

@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts.

    We load the embedding model here so the first search is fast. Without
    this it loads on the first search instead, and that user waits a few
    seconds for nothing.
    """
    if embeddings.enabled():
        print("Loading the embedding model...")
        embeddings.embed("warm up")
        print("Embedding model ready.")
    else:
        print("Embeddings are off. Search will use keywords only.")

    yield        # the server runs here


app = FastAPI(title="NoteFlow", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

conn = db.connect("noteflow.db")
db.init_schema(conn)

# One Agent per browser session, kept in memory.
# A real multi-user app would put this in Redis so several workers could
# share it and it would survive a restart. One user and one process here,
# so a plain dict is enough.
SESSIONS = {}


def get_agent(request: Request) -> tuple[str, Agent]:
    """Find this browser's agent, or make a new one."""
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())

    if session_id not in SESSIONS:
        SESSIONS[session_id] = Agent(conn)

    return session_id, SESSIONS[session_id]


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
async def chat(request: Request, response: Response):
    """Handle one message and return the agent's answer."""
    body = await request.json()
    message = (body.get("message") or "").strip()

    session_id, agent = get_agent(request)

    if not message:
        answer = {"reply": "Say something first.", "kind": "reply", "options": []}
    else:
        try:
            answer = agent.handle(message)
        except Exception as error:
            answer = {
                "reply": f"Something went wrong: {error}",
                "kind": "reply",
                "options": [],
            }

    response.set_cookie("session_id", session_id, httponly=True)
    return answer


@app.get("/notes")
def notes():
    """The sidebar list, so changes are visible straight away."""
    return {"notes": db.search_notes(conn, limit=50)}


@app.post("/seed")
def seed():
    """Add demo notes so the tricky cases can be shown quickly."""
    examples = [
        ("API migration", "Auth endpoint changes and the deadline.", ["engineering"]),
        ("API authentication", "Token refresh rules.", ["engineering"]),
        ("API documentation", "Rewrite the getting started page.", ["docs"]),
        ("Team standup", "We agreed to move it to Tuesdays.", ["meetings"]),
        ("Old office address", "123 Main Street, floor 2.", ["admin"]),
        ("اجتماع الفريق", "تم نقل الاجتماع الى يوم الثلاثاء.", ["اجتماعات"]),
        ("ملاحظات المشروع", "الموعد النهائي يوم الاثنين.", ["عمل"]),
    ]

    for title, body, tags in examples:
        db.add_note(conn, title, body, tags)

    return {"added": len(examples)}


@app.get("/health")
def health():
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    return {"ok": True, "api_key": has_key}
