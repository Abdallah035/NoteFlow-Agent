"""Search notes by meaning instead of by matching letters.

Keyword search cannot connect "فواتير" to "فاتورة". Embeddings can, because
they compare meaning.

Switched on with NOTEFLOW_EMBEDDINGS=1. Everything still works without it.
"""

import os

import numpy as np

from noteflow import vectorstore

# Read the model from the local cache so nothing is downloaded.
os.environ.setdefault("HF_HOME", r"F:\hf-cache")

# Small enough to run on a normal laptop, and one of its strongest
# languages is Arabic (71.4 on MIRACL, against 76.0 for the large model).
MODEL_NAME = os.getenv("NOTEFLOW_EMBED_MODEL", "intfloat/multilingual-e5-small")

model = None


def enabled():
    """True if embeddings are switched on."""
    return os.getenv("NOTEFLOW_EMBEDDINGS", "") == "1"


def embed(text, is_query=False):
    """Turn one piece of text into a vector.

    This model was trained with "query:" and "passage:" in front of the text,
    so we always add them. Leaving them out gives worse results.
    """
    global model

    if model is None:
        from sentence_transformers import SentenceTransformer
        # Downloads once, then loads from the cache every time after.
        model = SentenceTransformer(MODEL_NAME)

    prefix = "query: " if is_query else "passage: "
    vector = model.encode(prefix + text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


# A note shorter than this is embedded in one piece.
SHORT_NOTE = 800

# When we do split, we aim for pieces about this long.
CHUNK_SIZE = 500


def split_body(body):
    """Cut a long body into pieces, one idea each.

    Short notes stay whole - splitting them would only lose context.
    Long ones are cut on blank lines, so a piece is a paragraph or two
    rather than half a sentence.
    """
    if len(body) <= SHORT_NOTE:
        return [body]

    chunks = []
    current = ""

    for paragraph in body.split("\n\n"):
        if current and len(current) + len(paragraph) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = current + "\n\n" + paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(note, piece):
    """The text we embed for one piece of a note.

    Every piece carries the title and tags. Without them a piece like
    "the deadline is Monday" has no idea what it is about, and matches badly.
    """
    tags = ", ".join(note["tags"])
    return f"Title: {note['title']}\nTags: {tags}\n{piece}"


def save_embedding(conn, note):
    """Store the vectors for one note.

    A short note gives one chunk. A long note gives several, all sharing the
    same note_id, so we can put them back together when searching.

    They go to Chroma if NOTEFLOW_VECTORDB=1, otherwise to SQLite.
    """
    if not enabled():
        return

    texts = [chunk_text(note, piece) for piece in split_body(note["body"])]
    vectors = [embed(text) for text in texts]

    if vectorstore.enabled():
        vectorstore.save(note["id"], texts, vectors)
        return

    conn.execute("DELETE FROM chunks WHERE note_id = ?", (note["id"],))
    for text, vector in zip(texts, vectors):
        conn.execute(
            "INSERT INTO chunks (note_id, chunk_text, embedding) VALUES (?, ?, ?)",
            (note["id"], text, vector.tobytes()),
        )
    conn.commit()


# The model gives about 0.78 even to text with nothing in common, and
# about 0.87 to a good match, so every score sits in a narrow band.
FLOOR = 0.78
CEILING = 0.87


def stretch(score):
    """Spread the model's scores out over 0 to 1.

    Without this, an unrelated note would still score 0.78, and a perfect
    match only 0.87 - too close together to tell apart.
    """
    stretched = (score - FLOOR) / (CEILING - FLOOR)
    return max(0.0, min(1.0, stretched))


def semantic_scores(conn, query):
    """Score every note against the query by meaning.

    A long note has several chunks. We score each chunk, then keep the best
    one for each note, so a note is judged by its most relevant part and
    never appears in the results twice.

    Returns {note_id: score} where the score is between 0 and 1.
    """
    if not enabled() or not query.strip():
        return {}

    query_vector = embed(query, is_query=True)

    if vectorstore.enabled():
        raw = vectorstore.search(query_vector)
        return {note_id: stretch(score) for note_id, score in raw.items()}

    rows = conn.execute("SELECT note_id, embedding FROM chunks").fetchall()
    if not rows:
        return {}

    scores = {}
    for row in rows:
        chunk_vector = np.frombuffer(row["embedding"], dtype=np.float32)

        # Both vectors are normalised, so the dot product is the similarity.
        score = stretch(float(np.dot(query_vector, chunk_vector)))

        note_id = row["note_id"]
        if score > scores.get(note_id, 0.0):
            scores[note_id] = score

    return scores


def best_chunk(conn, note_id, query):
    """The piece of a note that best matches the query, for showing the user."""
    if not enabled():
        return None

    rows = conn.execute(
        "SELECT chunk_text, embedding FROM chunks WHERE note_id = ?", (note_id,)
    ).fetchall()
    if not rows:
        return None

    query_vector = embed(query, is_query=True)

    best = None
    best_score = -1.0
    for row in rows:
        chunk_vector = np.frombuffer(row["embedding"], dtype=np.float32)
        score = float(np.dot(query_vector, chunk_vector))
        if score > best_score:
            best_score = score
            best = row["chunk_text"]

    return best


def rebuild_all(conn):
    """Work out the vectors for every note. Run once after switching on."""
    from noteflow import db

    notes = db.search_notes(conn, limit=10000)
    for note in notes:
        save_embedding(conn, note)

    return len(notes)
