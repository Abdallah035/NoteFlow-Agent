"""The brain: one agent loop driven by the model.

The model calls a tool, we run it, we give the result back, and the model
decides what to say or do next. Nothing is written to the database until the
user presses Yes, because the writing tools are only unlocked by a
confirmation the user actually gave.
"""

import os
from datetime import date

from google import genai
from google.genai import types
from pydantic import ValidationError

from noteflow import db, embeddings
from noteflow.guard import build_diff
from noteflow.matcher import match
from noteflow.state import SessionState
from noteflow.tools import all_declarations, validate_call

MODEL = os.getenv("NOTEFLOW_MODEL", "gemma-4-31b-it")

MAX_STEPS = 4       # how many tool calls we allow in one turn
HISTORY_LIMIT = 12  # turns we send back. The free tier allows 16k input
                    # tokens per minute, so we keep the context small.

SYSTEM_PROMPT = """You are NoteFlow, a note taking assistant.

Today is {today}.

HOW YOU WORK
You never change anything without asking first. Every write is a two step
dance: you propose, the user accepts.

ADDING A NOTE
1. When the user gives you something to save, call preview_add with the title,
   body and tags you propose. This does NOT save anything.
2. The user then says yes or asks for a change.
3. If they ask for a change ("make the title shorter", "add tag work",
   "no, the body should say Monday"), call preview_add again with the fixed
   values. Keep doing this until they accept.
4. Only after the user clearly agrees, call add_note with the final values.

If you are not sure the user even wants a note saved, ask them first in plain
text. Do not call preview_add for a question or for small talk.

UPDATING OR DELETING
1. First call search_notes to find candidates. Never guess which note.
2. Look at the result. Show the user what you found and ask which one they
   mean, unless there is only one and it obviously matches.
3. When you know the note, call preview_update or preview_delete with the
   note id. This does NOT change anything.
4. Only after the user clearly agrees, call update_note or delete_note.

DELETING EVERYTHING
If the user asks to wipe all their notes, call preview_delete_all. It warns
them and asks them to type DELETE ALL.
Then call delete_all_notes and put EXACTLY what they typed into
confirm_phrase. A plain "yes" is not enough for this one, so if they only say
yes, tell them they need to type DELETE ALL.
If they want only some notes gone, do not use this - search and delete those
notes one at a time instead.

LISTING
Call search_notes with no filters to list everything. After showing a list,
offer to update or delete one of them.

CHANGING THEIR MIND
The user can change direction at any moment. If they were adding a note and
then ask to see their notes, just do that instead. If they cancel, say
something friendly and ask what else they need. Never carry on with an action
they have dropped.

STYLE
Reply in the same language the user writes in. Keep answers short. When you
list notes, number them so the user can say "the second one".
Only talk about notes. If asked anything else, say kindly that you only
handle notes.
"""


def reply(text, kind="reply", options=None):
    """One answer from the agent. 'kind' tells the UI how to draw it."""
    return {"reply": text, "kind": kind, "options": options or []}


def friendly_error(error):
    """Turn an API error into something the user can understand."""
    text = str(error)

    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "I am busy right now. Please wait a few seconds and try again."
    if "503" in text or "UNAVAILABLE" in text:
        return "The model is overloaded right now. Please try again."

    return f"Sorry, something went wrong: {text[:150]}"


class Agent:
    """Handles one conversation."""

    def __init__(self, conn, api_key=None):
        self.conn = conn
        self.state = SessionState()
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

    # ---------- the main loop ----------

    def handle(self, message: str) -> dict:
        """Handle one user message.

        We ask the model what to do. If it picks a tool we run it, give the
        answer back, and ask again. We stop when the model just talks, or
        when a tool needs the user to answer something.
        """
        self.remember("user", message)
        buttons = []

        for _ in range(MAX_STEPS):
            try:
                response = self.ask_model()
            except Exception as error:
                return reply(friendly_error(error))

            call = self.get_tool_call(response)

            # No tool call means the model just wants to talk.
            if call is None:
                text = (response.text or "").strip()
                self.remember("model", text)
                kind = "clarify" if buttons else "reply"
                return reply(text or "How can I help with your notes?",
                             kind, buttons)

            # Keep the model's own turn, or it forgets what it just did.
            self.state.history.append(response.candidates[0].content)

            result, new_buttons, question = self.run_tool(call.name,
                                                          dict(call.args))
            self.tell_model_the_result(call.name, result)

            if new_buttons:
                buttons = new_buttons

            # A question means we need the user before going further.
            if question is not None:
                self.remember("model", question)
                kind = "confirm" if call.name.startswith("preview_") else "clarify"
                return reply(question, kind, buttons)

        return reply("That took too many steps. Could you say it more simply?")

    def ask_model(self):
        """Send the conversation and the tool list to the model."""
        return self.client.models.generate_content(
            model=MODEL,
            contents=self.state.history[-HISTORY_LIMIT:],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.format(today=date.today()),
                tools=[types.Tool(function_declarations=all_declarations())],
            ),
        )

    def get_tool_call(self, response):
        """Pull the first tool call out of the model's reply, or None."""
        parts = response.candidates[0].content.parts or []
        for part in parts:
            if getattr(part, "function_call", None):
                return part.function_call
        return None

    def tell_model_the_result(self, tool_name: str, result: dict):
        """Put the tool's answer into the history so the model can see it."""
        self.state.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=tool_name, response={"result": result})],
            )
        )

    # ---------- running one tool ----------

    def run_tool(self, name: str, args: dict):
        """Run one tool and return what happened.

        Every tool returns three things:
          result   - what we tell the model
          buttons  - what the user can click
          question - text to show the user now, or None to let the model
                     carry on. A question means we stop and wait for them.
        """
        try:
            args = validate_call(name, args).model_dump()
        except (ValidationError, ValueError) as error:
            return {"error": str(error)[:200]}, [], None

        if name == "search_notes":
            return self.do_search(args)
        if name == "preview_add":
            return self.do_preview_add(args)
        if name in ("preview_update", "preview_delete"):
            return self.do_preview_change(name, args)
        if name == "preview_delete_all":
            return self.do_preview_delete_all(args)
        if name == "add_note":
            return self.do_add(args)
        if name == "update_note":
            return self.do_update(args)
        if name == "delete_note":
            return self.do_delete(args)
        if name == "delete_all_notes":
            return self.do_delete_all(args)

        return {"error": "unknown tool"}, [], None

    # ---------- reads ----------

    def do_search(self, args):
        """Find notes. The model decides what to say about them."""
        query = (args.get("query") or "").strip()

        # Tags and dates are filtered in SQL. The keyword ranking happens in
        # Python, so we ask for everything that passes the filters.
        notes = db.search_notes(
            self.conn,
            tags=args.get("tags") or None,
            date_from=str(args["date_from"]) if args.get("date_from") else None,
            date_to=str(args["date_to"]) if args.get("date_to") else None,
            limit=50,
        )

        if query:
            meaning = embeddings.semantic_scores(self.conn, query)
            outcome, found = match(query, notes, semantic_scores=meaning)
            if outcome == "none":
                found = []
        else:
            found = notes[: args.get("limit", 10)]

        self.state.last_results = found
        if found:
            self.state.last_note_id = found[0]["id"]

        # Short version for the model, so we do not waste tokens on long bodies.
        summary = [
            {"id": n["id"], "title": n["title"], "body": n["body"][:100],
             "tags": n["tags"], "created": n["created_at"][:10]}
            for n in found
        ]
        return {"count": len(found), "notes": summary}, self.note_buttons(found), None

    # ---------- previews: propose, never write ----------

    def do_preview_add(self, args):
        """Show the note we would save and wait for a yes."""
        self.state.pending_action = {"tool": "add_note", "args": args}

        tags = ", ".join(args.get("tags") or []) or "none"
        text = (
            f"Here is the note I will save:\n\n"
            f"Title: {args['title']}\n"
            f"Body: {args['body']}\n"
            f"Tags: {tags}\n\n"
            f"Shall I save it?"
        )
        return {"shown_to_user": True}, self.yes_no(), text

    def do_preview_change(self, name, args):
        """Show what an update or delete would do, and wait for a yes."""
        note = self.find_note(args.get("note_ref", ""))
        if note is None:
            return {"error": "no note matched, ask the user which one"}, [], None

        self.state.last_note_id = note["id"]

        if name == "preview_delete":
            self.state.pending_action = {"tool": "delete_note", "note": note}
            question = (f"Delete note {note['id']} '{note['title']}'?\n\n"
                        f"This cannot be undone.")
        else:
            self.state.pending_action = {"tool": "update_note",
                                         "args": args, "note": note}
            changes = "\n".join(build_diff(note, args))
            question = f"Update note {note['id']} '{note['title']}'?\n\n{changes}"

        return {"shown_to_user": True}, self.yes_no(), question

    def do_preview_delete_all(self, args):
        """Warn about deleting everything and ask for a typed word."""
        total = db.count_notes(self.conn)

        if total == 0:
            return {"count": 0, "note": "there is nothing to delete"}, [], None

        self.state.pending_action = {"tool": "delete_all_notes"}

        titles = "\n".join(
            f"  - {n['title']}" for n in db.search_notes(self.conn, limit=5)
        )
        more = f"\n  ...and {total - 5} more" if total > 5 else ""

        text = (
            f"This will delete ALL {total} of your notes:\n\n"
            f"{titles}{more}\n\n"
            f"This cannot be undone.\n"
            f"Type DELETE ALL to confirm, or press Cancel."
        )
        # No Yes button on purpose: a slip of the finger must not wipe
        # everything. The user has to type the words.
        return ({"shown_to_user": True, "count": total},
                [{"label": "Cancel", "value": "cancel"}],
                text)

    # ---------- writes: only reachable after a preview ----------

    def do_add(self, args):
        """Save a note, but only if it was previewed first."""
        pending = self.state.pending_action
        if not pending or pending["tool"] != "add_note":
            return self.do_preview_add(args)

        note_id = db.add_note(self.conn, args["title"], args["body"],
                              args.get("tags") or [])
        self.state.clear_pending()
        self.state.last_note_id = note_id
        return {"saved": True, "id": note_id, "title": args["title"]}, [], None

    def do_update(self, args):
        pending = self.state.pending_action
        if not pending or pending["tool"] != "update_note":
            return self.do_preview_change("preview_update", args)

        note = pending["note"]
        updated = db.update_note(
            self.conn, note["id"],
            title=args.get("title"),
            body=args.get("body"),
            tags=args.get("tags"),
            body_mode=args.get("body_mode", "append"),
            tags_mode=args.get("tags_mode", "append"),
        )
        self.state.clear_pending()
        return {"updated": True, "id": note["id"], "title": updated["title"]}, [], None

    def do_delete_all(self, args):
        """Delete everything, but only after the user typed the exact words."""
        pending = self.state.pending_action
        if not pending or pending["tool"] != "delete_all_notes":
            return self.do_preview_delete_all({})

        # We check the phrase ourselves. The model is not allowed to decide
        # that "yes" was good enough.
        typed = str(args.get("confirm_phrase", "")).strip().upper()
        if typed not in ("DELETE ALL", "DELETEALL"):
            return ({"error": "the user did not type DELETE ALL, so nothing "
                              "was deleted. Ask them to type it exactly."},
                    [{"label": "Cancel", "value": "cancel"}], None)

        removed = db.delete_all_notes(self.conn)
        self.state.clear_pending()
        self.state.last_note_id = None
        self.state.last_results = []
        return {"deleted_all": True, "count": removed}, [], None

    def do_delete(self, args):
        pending = self.state.pending_action
        if not pending or pending["tool"] != "delete_note":
            return self.do_preview_change("preview_delete", args)

        note = pending["note"]
        deleted = db.delete_note(self.conn, note["id"])
        self.state.clear_pending()
        self.state.last_note_id = None
        return {"deleted": True, "title": deleted["title"]}, [], None

    # ---------- small helpers ----------

    def find_note(self, note_ref: str):
        """Turn what the user said into one note, or None."""
        note_ref = str(note_ref).strip()

        vague = {"that", "it", "this", "that note", "the last one", "the note",
                 "تلك", "هذه", "الاخيره", "الأخيرة"}
        if note_ref.lower() in vague and self.state.last_note_id:
            note_ref = str(self.state.last_note_id)

        notes = db.search_notes(self.conn, limit=50)
        meaning = embeddings.semantic_scores(self.conn, note_ref)
        outcome, candidates = match(note_ref, notes, semantic_scores=meaning)

        if outcome == "one":
            return candidates[0]
        return None

    def note_buttons(self, notes):
        """Buttons so the user can pick a note by clicking."""
        return [
            {"label": f"{i}. {n['title']}", "value": f"the note titled {n['title']}"}
            for i, n in enumerate(notes[:5], start=1)
        ]

    def yes_no(self):
        return [{"label": "Yes", "value": "yes"},
                {"label": "No", "value": "no"}]

    def remember(self, who: str, text: str):
        """Add one line to the conversation history."""
        if not text:
            return
        role = "user" if who == "user" else "model"
        self.state.history.append(
            types.Content(role=role, parts=[types.Part(text=text)])
        )
