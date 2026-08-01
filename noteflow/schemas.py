"""Tool schemas — the contract between the LLM and our database.

Every docstring and description below is read by the model before it chooses
a tool, so they are written as instructions to the model, not as notes to us.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AddNote(BaseModel):
    """Actually save a note, AFTER the user has agreed to a preview.

    Never call this first. Call preview_add first and wait for the user to
    say yes. Only then call this with the exact values they accepted.

    Do NOT use this to change an existing note — use preview_update.
    Do NOT use this for questions about saved notes — use search_notes.
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description=(
            "A short title, 3-8 words. If the user gave a title, use their exact "
            "words. If they did not, write one that summarises the body. Never "
            "copy the whole body into the title. Never invent facts that are not "
            "in the body. Write the title in the SAME language the user wrote in."
        ),
    )
    body: str = Field(
        ...,
        min_length=1,
        description=(
            "The note content in the user's own words. Keep their meaning and "
            "details exactly - do not summarise, shorten, or translate. Remove "
            "only the command part: from 'save a note that the standup moved to "
            "Tuesday', the body is 'The standup moved to Tuesday.'"
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Topic labels, lowercase, single words where possible, e.g. "
            "['meetings'] or ['work','urgent']. Add a tag ONLY if the user asked "
            "for one ('tag it as X') or the topic is obvious. Prefer 0-3 tags. "
            "Return [] when unsure - a wrong tag is worse than no tag."
        ),
    )


class SearchNotes(BaseModel):
    """Find, list, or look up existing notes.

    Use for: "what did I write about X", "show my notes", "find the API note",
    "any notes from last week", or any question about saved content.

    Also use this FIRST when the user wants to update or delete a note but you
    are not sure which note they mean.

    Leave every field empty to list all notes, newest first.
    """

    query: str | None = Field(
        None,
        description=(
            "The key words to search for, not the whole sentence. From 'what did "
            "I write about the API migration last week?' the query is 'API "
            "migration' - the date belongs in date_from/date_to. Keep the user's "
            "original language; do not translate. Leave empty to list everything."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Filter by tag. Use ONLY when the user names a tag or category "
            "explicitly ('my meeting notes', 'tagged urgent'). A note must have "
            "ALL listed tags to match, so pass one tag unless they asked for more."
        ),
    )
    date_from: date | None = Field(
        None,
        description=(
            "Earliest date as YYYY-MM-DD. Convert relative words yourself using "
            "today's date given in the system prompt: 'last week' -> the Monday "
            "7 days back, 'yesterday' -> today minus 1, 'in July' -> 2026-07-01. "
            "Leave empty when the user gave no time hint."
        ),
    )
    date_to: date | None = Field(
        None,
        description=(
            "Latest date as YYYY-MM-DD, inclusive. Set together with date_from "
            "for a range ('last week', 'between May and June'). Leave empty for "
            "'since X' style requests."
        ),
    )
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description=(
            "How many notes to return. Keep the default 10. Raise it only if the "
            "user asks for 'all' or names a number."
        ),
    )


class PreviewAdd(BaseModel):
    """Show the user the note you propose to save, and ask them to confirm.

    This does NOT save anything. It only shows the note and asks.

    Call this FIRST whenever the user gives you something to remember.
    If they then ask for a change ("shorter title", "add tag work"), call
    this again with the corrected values. Repeat until they agree.

    Only after the user clearly agrees do you call add_note.
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description=(
            "The title you propose, 3-8 words, in the user's language. Use "
            "their words if they gave a title."
        ),
    )
    body: str = Field(
        ...,
        min_length=1,
        description=(
            "The note content in the user's own words, without the command "
            "part. Do not summarise or translate it."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Lowercase topic labels. Use [] when unsure.",
    )


class PreviewUpdate(BaseModel):
    """Show the user what an update would change, and ask them to confirm.

    This does NOT change anything.

    Call search_notes first so you know which note you mean. Only call this
    once you have a real note id. After the user agrees, call update_note.
    """

    note_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "The id of the note, as a number, taken from a previous "
            "search_notes result. Only use words if you have no id."
        ),
    )
    title: str | None = Field(None, max_length=120,
                              description="New title, or leave empty to keep it.")
    body: str | None = Field(
        None,
        description="Text to add or to replace, or leave empty to keep it.",
    )
    body_mode: Literal["append", "replace"] = Field(
        "append",
        description=(
            "'append' adds and keeps the old text. 'replace' erases it. "
            "Choose append when unsure."
        ),
    )
    tags: list[str] | None = Field(None, description="Tags to add or set.")
    tags_mode: Literal["append", "replace"] = Field(
        "append", description="'append' keeps the old tags, 'replace' drops them."
    )


class PreviewDelete(BaseModel):
    """Show the user which note would be deleted, and ask them to confirm.

    This does NOT delete anything.

    Call search_notes first. Only call this once you know exactly which note
    the user means. After the user agrees, call delete_note.
    """

    note_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "The id of the note, as a number, taken from a previous "
            "search_notes result."
        ),
    )


class PreviewDeleteAll(BaseModel):
    """Warn the user that ALL their notes would be deleted, and ask them.

    This does NOT delete anything.

    Use only when the user clearly asks to remove everything: "delete all my
    notes", "clear everything", "احذف كل الملاحظات".

    If they only want some notes gone ("delete my API notes"), do NOT use
    this - search first and delete them one by one.
    """

    reason: str = Field(
        "",
        description=(
            "Optional: what the user said they wanted, in a few words. "
            "Leave empty if they simply asked to delete everything."
        ),
    )


class DeleteAllNotes(BaseModel):
    """Actually delete EVERY note. This cannot be undone.

    Never call this first, and never call it just because the user said
    "yes". Call preview_delete_all, and only call this after the user has
    typed the exact word the preview asked them for.
    """

    confirm_phrase: str = Field(
        ...,
        description=(
            "The exact word the user typed to confirm. Copy what they typed, "
            "letter for letter. Never invent it or fill it in yourself."
        ),
    )


class UpdateNote(BaseModel):
    """Actually change a note, AFTER the user has agreed to a preview.

    Never call this first. Call preview_update, wait for the user to say yes,
    then call this with the same values.
    """

    note_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "How the user referred to the note, in their words: 'the standup "
            "note', 'API migration', 'the last one', or a plain number if they "
            "gave an id. Copy their wording - do NOT guess a database id and do "
            "NOT invent a title you have not seen."
        ),
    )
    title: str | None = Field(
        None,
        max_length=120,
        description=(
            "New title. Set ONLY if the user asked to change the title. "
            "Leave empty to keep the current one."
        ),
    )
    body: str | None = Field(
        None,
        description=(
            "The new text. With body_mode='append' send ONLY the sentence to "
            "add, not the old text repeated. Leave empty to keep the body as is."
        ),
    )
    body_mode: Literal["append", "replace"] = Field(
        "append",
        description=(
            "'append' adds your text to the end and keeps what is already there "
            "- use it for 'add', 'also', 'include'. 'replace' erases the old "
            "body - use it ONLY when the user clearly says 'replace', 'rewrite', "
            "or 'change it to say'. When unsure choose 'append', because append "
            "loses nothing."
        ),
    )
    tags: list[str] | None = Field(
        None,
        description=(
            "Tags to add or set. Leave empty to keep the current tags. "
            "Send [] with tags_mode='replace' to clear all tags."
        ),
    )
    tags_mode: Literal["append", "replace"] = Field(
        "append",
        description=(
            "'append' keeps existing tags and adds yours - the normal choice for "
            "'tag it as X'. 'replace' throws the old tags away; use only when "
            "the user says 'change the tags to' or asks to remove tags."
        ),
    )


class DeleteNote(BaseModel):
    """Actually delete a note, AFTER the user has agreed to a preview.

    Never call this first. Call preview_delete, wait for the user to say yes,
    then call this. It cannot be undone.
    """

    note_ref: str = Field(
        ...,
        min_length=1,
        description=(
            "How the user referred to the note, in their words: 'the old office "
            "address', 'that one', or a plain number if they gave an id. Copy "
            "their wording - never guess a database id, and never fall back to "
            "the most recent note just because you are unsure."
        ),
    )
