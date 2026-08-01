# NoteFlow

A conversational note-taking agent. You manage your notes by talking to it,
in English or Arabic.

```
you:   Save a note about the team standup, we moved it to Tuesdays
agent: Here is the note I will save:
       Title: Team standup
       Body: The standup moved to Tuesdays.
       Tags: meetings
       Shall I save it?                          [Yes] [No]

you:   delete my API note
agent: I found 3 notes. Which one do you mean?
       [1. API migration] [2. API authentication] [3. API documentation]
```

## Setup

```bash
uv sync                                  # install
cp .env.example .env                     # then put your API key in .env
```

Get a free key at https://aistudio.google.com/apikey

## Run

```bash
uv run uvicorn noteflow.app:app --reload     # web UI at http://127.0.0.1:8000
uv run python -m noteflow.cli                # or the terminal version
```

## How it works

```
your message
     |
Orchestrator  ------ asks the model, runs the tool it picked,
     |                gives the result back, asks again
   Matcher     ------ works out which note "the standup one" means
     |
Safety guard  ------ nothing is written until you say yes
     |
   SQLite
```

### Tools

The model gets nine tools. They come in pairs on purpose:

| Proposes (safe) | Carries out |
|---|---|
| `preview_add` | `add_note` |
| `preview_update` | `update_note` |
| `preview_delete` | `delete_note` |
| `preview_delete_all` | `delete_all_notes` |

Plus `search_notes`.

The preview tools cannot touch the database. They only show you what would
happen. The writing tools check that a preview was shown first, so even if
the model ignores its instructions, nothing is saved without your approval.

Each tool is a Pydantic model in `schemas.py`. The same model both describes
the tool to the LLM and validates what comes back, so the two can never drift
apart.

### Three decisions worth explaining

**`note_ref` is a string, not an id.** The model passes back your own words
("the standup note"), never a database id. If it were an id the model would
invent one and delete the wrong note. A string means our own matcher resolves
it, and can say "I found three, which one?" instead of guessing.

**Append is the default.** "Add a deadline to that note" must never erase the
body. The database layer defaults to replace because that is what update
plainly means; the tool layer defaults to append because a model that is
unsure should add text, not lose it.

**Deleting everything needs typing, not clicking.** There is no Yes button
for `delete_all_notes`. You have to type `DELETE ALL`, and Python checks the
phrase, not the model. A misclick cannot wipe your notes.

### Finding the right note

Words first, meaning second.

`matcher.py` scores every note on its words, from 0 to 1:

| Signal | Weight |
|---|---|
| Exact title match | 35% |
| Title keyword match | 20% |
| Body keyword match | 15% |
| Meaning (embeddings) | 20% |
| Recency | 10% |

If nothing passes 0.35, it tries again using meaning alone. That is how
"فواتير" finds a note titled "فاتورة", and how "delayed deployment" finds one
that says "the release was postponed" - no shared words at all.

Keyword matches are precise, so when they work we trust them and never reach
the fallback. A note matching the words exactly still beats one that only
matches in meaning.

Then it decides:

- one note well above the rest -> use it
- several close scores -> ask which one
- nothing found either way -> say so

Scores of 0.80, 0.78 and 0.77 are noise, not a preference, so anything within
0.10 of the winner counts as a tie and the agent asks. That is the rule that
stops it deleting the wrong note.

With embeddings switched off the weights would only add up to 0.80, so a
perfect match could never pass the 0.75 mark. The missing 20% is shared out
over the other signals instead.

### Arabic

- spelling variants are normalised: أ إ آ -> ا, ة -> ه, ى -> ي
- diacritics and tatweel are stripped
- "ال" is removed from the front of words, so السداد matches سداد
- tokens use `\w+`, not `[a-z]+`, which would silently return nothing

Broken plurals (فاتورة -> فواتير) are left to the embeddings, because letter
rules for them either miss cases or match unrelated words.

## Semantic search (optional)

```bash
uv sync --extra embeddings
export NOTEFLOW_EMBEDDINGS=1
uv run python -c "from noteflow import db, embeddings; c=db.connect('noteflow.db'); db.init_schema(c); embeddings.rebuild_all(c)"
```

Uses `intfloat/multilingual-e5-small` locally, so no API calls and no quota.
It downloads once (about 470MB) and reads from the cache after that.

A short note gets one vector. A long note is cut into pieces on blank lines,
so each piece is about one idea, and all the pieces share the same note id.
When searching, we score every piece and keep the best one per note, so a
note is judged by its most relevant part and never appears twice.

Every piece carries the note's title and tags, because a piece saying "the
deadline is Monday" is meaningless without knowing what it belongs to.

The model's raw scores sit in a narrow band - about 0.78 for unrelated text
and 0.87 for a good match - so `stretch()` spreads that range over 0 to 1.

## Layout

```
noteflow/
  db.py             SQLite: CRUD and the audit log
  schemas.py        the tool definitions the model reads
  tools.py          turns those into the API's format
  matcher.py        which note did they mean
  guard.py          describes a change before you accept it
  orchestrator.py   the agent loop
  embeddings.py     semantic search
  state.py          what one conversation remembers
  app.py            FastAPI web UI
  cli.py            terminal version
  static/           the chat page: html, css, one js file
```

## Choices and limits

**SQLite, not Postgres.** No server to install, so the whole project runs
from a clone. The database layer is one file, so swapping it is contained.

**Sessions in a dict, not Redis.** One user, one process. Restarting loses
the conversation. `SessionState` is a plain dataclass with no server logic
in it, so moving it to Redis is one function.

**No LangGraph.** The routing is already a small state machine you can read
in three lines. A framework would hide the tool loop, which is the part worth
showing.

**Notes are hard deleted.** The audit log keeps the content, so nothing is
truly lost, and no query has to remember a `WHERE deleted_at IS NULL`.

**Tags are JSON in a column.** A join table would be more correct but costs
two tables and a join for a feature that holds a handful of labels.

## Environment

| Variable | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | - | required |
| `NOTEFLOW_MODEL` | `gemma-4-31b-it` | any Gemini or Gemma model |
| `NOTEFLOW_EMBEDDINGS` | off | `1` turns on semantic search |
