<div align="center">

# NoteFlow

**A conversational note-taking agent — manage your notes by talking to it, in English or Arabic.**

</div>

---

```
you:   Save a note about the team standup, we moved it to Tuesdays

agent: Here is the note I will save:

         Title: Team standup
         Body:  The standup moved to Tuesdays.
         Tags:  meetings

       Shall I save it?                            [ Yes ]  [ No ]

you:   delete my API note

agent: I found 3 notes. Which one do you mean?

       [ 1. API migration ]  [ 2. API authentication ]  [ 3. API documentation ]

you:   the second one

agent: Delete note 2 'API authentication'?
       This cannot be undone.                      [ Yes ]  [ No ]
```

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Tool design](#tool-design)
- [Conversation state](#conversation-state)
- [Finding the right note](#finding-the-right-note)
- [Arabic support](#arabic-support)
- [Semantic search](#semantic-search)
- [Project layout](#project-layout)
- [Decisions and limits](#decisions-and-limits)

---

## Quick start

```bash
uv sync                        # install
cp .env.example .env           # then put your API key in .env
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
uv run uvicorn noteflow.app:app --reload    # web UI at http://127.0.0.1:8000
uv run python -m noteflow.cli               # or the terminal version
```

Optional semantic search:

```bash
uv sync --extra embeddings     # adds the local embedding model
# then set NOTEFLOW_EMBEDDINGS=1 in .env
```

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | — | required |
| `NOTEFLOW_MODEL` | `gemma-4-31b-it` | any Gemini or Gemma model with tool support |
| `NOTEFLOW_EMBEDDINGS` | off | `1` turns on semantic search |
| `NOTEFLOW_VECTORDB` | off | `1` uses Qdrant instead of SQLite for vectors |

---

## Architecture

![Overall architecture](Architecture%20images/Overall%20Architecture.png)

A message travels through five stages. Each one has a single job, and each one
can hand control back to the user instead of guessing.

| Stage | File | Job |
|---|---|---|
| **1. Orchestrator** | `orchestrator.py` | Runs the agent loop: ask the model, run the tool it picked, feed the result back, ask again |
| **2. Note Matcher** | `matcher.py` | Turns "the standup note" into a real note id, or decides it cannot tell. Keywords first, embeddings as a fallback |
| **3. Safety Guard** | `guard.py` + tool split | Nothing is written until the user says yes |
| **4. Tool Executor** | `tools.py` | Validates the model's arguments before anything runs |
| **5. Data Layer** | `db.py` | SQLite: notes, audit log, embedding chunks |

The loop is what makes it an agent rather than a command parser: the model sees
each tool's result and decides what to do next, so the user can change their
mind mid-flow and it keeps up.

---

## Tool design

![Tool contract](Architecture%20images/Tools%20schema.png)

The model gets **nine tools**, deliberately split into pairs:

| Proposes — cannot write | Carries out |
|---|---|
| `preview_add` | `add_note` |
| `preview_update` | `update_note` |
| `preview_delete` | `delete_note` |
| `preview_delete_all` | `delete_all_notes` |

Plus `search_notes`, which is read-only and never needs confirmation.

Preview tools **cannot touch the database**. They only show what would happen.
The writing tools check that a matching preview was shown first, so even if the
model ignores its instructions, nothing is saved without approval.

Each tool is a Pydantic model in `schemas.py`. The same model both describes the
tool to the LLM and validates what comes back, so the two can never drift apart.

### Three decisions worth explaining

**`note_ref` is a string, not an id.**
The model passes back the user's own words — `"the standup note"` — never a
database id. If it were an `int`, the model would invent one and delete the
wrong note. A string means our own matcher resolves it, and can answer
*"I found three, which one?"* instead of guessing. This single choice is what
makes ambiguity detection possible at all.

**Append is the default.**
*"Add a deadline to that note"* must never erase the body. The database layer
defaults to `replace`, because that is what "update" plainly means. The tool
layer defaults to `append`, because a model that is unsure should add text,
not lose it. Different layers, different defaults, on purpose.

**Deleting everything needs typing, not clicking.**
There is no Yes button for `delete_all_notes`. The user must type `DELETE ALL`,
and **Python** checks the phrase — not the model. A misclick cannot wipe the
notes, and the model cannot decide that "yes" was close enough.

---

## Conversation state

![Conversation state machine](Architecture%20images/State%20Architecture.png)

What one session remembers, in `state.py`:

| Field | Purpose |
|---|---|
| `last_note_id` | resolves "that note" |
| `last_results` | resolves "the second one" |
| `pending_action` | a change shown to the user, waiting for yes or no |
| `history` | the conversation, sent back to the model for context |

`pending_action` is the safety latch. The writing tools refuse to run unless it
matches what they were asked to do, so a write can only ever follow a preview
the user actually saw.

> **Note on the diagram:** it shows a separate `pending_clarify` state. In the
> final build the model handles clarification itself — it sees the search
> results and asks which note. That removed a whole branch of hand-written
> routing, so the code has one pending slot instead of two.

---

## Finding the right note

**Words first, meaning second.** Two passes, not one blended score.

### Pass 1 — keywords

Every note is scored from 0 to 1 on its words:

| Signal | Weight |
|---|---|
| Exact title match | 45% |
| Title keyword match | 25% |
| Body keyword match | 20% |
| Recency | 10% |

Anything scoring `0.30` or above is a candidate.

One guard: if neither the title nor the body matched a single word, the score is
**0**, whatever the note's age. Recency is there to break ties between real
matches, not to turn a fresh note into one.

### Pass 2 — meaning, only if pass 1 found nothing

If no note passes, we search again on embedding similarity alone. That is how
`فواتير` finds a note titled `فاتورة`, and how *"delayed deployment"* finds one
saying *"the release was postponed"* — no shared words at all.

**Why a fallback and not one blended score.** If meaning were just another
weighted signal, a note found only by meaning would score far below the floor
and be thrown away. Running the passes separately means keyword matches keep
their precision, and meaning only speaks when words have nothing to say.

### The decision

| Outcome | When | The agent |
|---|---|---|
| **one** | a clear winner above 0.75 | uses it |
| **multiple** | close scores | asks which one |
| **none** | nothing found in either pass | says so, suggests refining |

Scores of 0.80, 0.78 and 0.77 are noise, not a preference. Anything within 0.10
of the winner counts as a tie, and the agent asks. **That is the rule that stops
it deleting the wrong note.**

---

## Arabic support

The whole app works in both languages, including cross-lingual search.

| Problem | Fix |
|---|---|
| `أ إ آ` vs `ا`, `ة` vs `ه`, `ى` vs `ي` | normalised to one form |
| Diacritics and tatweel (`مــلاحظة`) | stripped |
| `ال` glued to the front of words | removed, so `السداد` matches `سداد` |
| `[a-z]+` silently returns nothing | tokens use `\w+` with Unicode |

That last one is the trap: with the wrong regex, every Arabic note tokenises to
an empty list and scores zero — no error, just search that never works. There is
a test guarding it.

Broken plurals (`فاتورة` → `فواتير`) are left to the embeddings, because letter
rules for them either miss real cases or match unrelated words.

---

## Semantic search

Optional, off by default. Uses `intfloat/multilingual-e5-small` **locally** —
no API calls, no quota, ~470MB downloaded once.

**Chunking.** A short note gets one vector. A long note is cut on blank lines so
each piece is about one idea, and every piece shares the same `note_id`. When
searching, each piece is scored and the best one per note wins — so a note is
judged by its most relevant part and never appears twice in the results.

Every piece carries the note's title and tags, because a chunk saying *"the
deadline is Monday"* is meaningless without knowing what it belongs to.

**Score stretching.** The model's raw scores sit in a narrow band — about 0.78
for unrelated text, 0.87 for a good match. `stretch()` maps that range onto
0–1 so the signal is usable.

Chunking runs inside `db.py`, right after the insert, so every path gets it —
the agent, the seed button, scripts. It cannot be forgotten.

---

## Project layout

```
noteflow/
  orchestrator.py   the agent loop
  schemas.py        tool definitions the model reads
  tools.py          converts them to the API's format
  matcher.py        which note did they mean
  guard.py          describes a change before you accept it
  db.py             SQLite: CRUD, audit log, chunk storage
  embeddings.py     semantic search
  vectorstore.py    optional Qdrant backend
  state.py          what one conversation remembers
  app.py            FastAPI web UI
  cli.py            terminal version
  static/           the chat page: html, css, one js file
```

---

## Decisions and limits

**SQLite, not Postgres.** No server to install, so the project runs from a
clone. The database layer is one file, so swapping it is contained.

**Sessions in a dict, not Redis.** One user, one process. Restarting loses the
conversation. `SessionState` is a plain dataclass with no server logic in it,
so moving it to Redis is one function.

**No LangGraph.** The routing is a small state machine that reads in three
lines. A framework would hide the tool loop, which is the part worth showing.

**Vectors in SQLite, not a vector database.** Comparing a few hundred normalised
vectors with numpy takes under a millisecond, while the LLM call in the same
request takes 500–2000ms — so the search is not the bottleneck. `vectorstore.py`
holds a working Qdrant adapter behind a flag to prove the swap is contained to
one file. At roughly 10k vectors I would switch, and I would pick **pgvector**,
so notes and vectors stay in one database and one transaction instead of keeping
two stores in sync.

**Notes are hard deleted.** The audit log keeps the content, so nothing is truly
lost, and no query has to remember a `WHERE deleted_at IS NULL`.

**Tags are JSON in a column.** A join table would be more correct but costs two
tables and a join for a feature that holds a handful of labels.

**Rate limits.** The free tier allows 16k input tokens per minute, so history is
trimmed to the last 12 turns and a 429 is reported as "please wait" rather than
crashing the turn.
