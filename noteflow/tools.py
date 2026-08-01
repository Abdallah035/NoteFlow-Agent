"""Turn Pydantic models into tool declarations the LLM understands.

Pydantic writes JSON Schema in its own style. The Gemini API accepts only a
subset of it, so we strip the keys it rejects and flatten optional fields.
"""

from google.genai import types

from noteflow.schemas import (
    AddNote,
    DeleteAllNotes,
    DeleteNote,
    PreviewAdd,
    PreviewDelete,
    PreviewDeleteAll,
    PreviewUpdate,
    SearchNotes,
    UpdateNote,
)

# The only keys Gemini wants. Pydantic adds several more that it rejects,
# such as "title", "default", "minLength" and "$defs".
_KEEP = {"type", "description", "enum", "items"}


def _clean(field: dict) -> dict:
    """Keep only the keys Gemini understands."""
    cleaned = {}

    for key, value in field.items():
        if key not in _KEEP:
            continue
        if key == "items":
            value = _clean(value)      # a list's item type is a schema too
        cleaned[key] = value

    return cleaned


def _unwrap_optional(field: dict) -> dict:
    """Turn Pydantic's anyOf[X, null] into plain X.

    Writing 'query: str | None' makes Pydantic say "string OR null", which
    Gemini rejects. We take the real type and drop the null. The field is
    still optional because it is not listed in "required".
    """
    if "anyOf" not in field:
        return field

    for option in field["anyOf"]:
        if option.get("type") != "null":
            real_type = dict(option)
            if "description" in field:
                real_type["description"] = field["description"]
            return real_type

    return field


def to_gemini_declaration(model, name: str) -> types.FunctionDeclaration:
    """Build one tool declaration from a Pydantic model.

    The model's docstring becomes the tool description the LLM reads when
    deciding which tool to call.
    """
    schema = model.model_json_schema()

    properties = {}
    for field_name, field_schema in schema.get("properties", {}).items():
        properties[field_name] = _clean(_unwrap_optional(field_schema))

    return types.FunctionDeclaration(
        name=name,
        description=(model.__doc__ or "").strip(),
        parameters={
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        },
    )


# The one place that says which tools exist. Everything else reads this.
# The preview tools propose a change; the plain ones carry it out once the
# user has agreed.
TOOLS = {
    "search_notes": SearchNotes,
    "preview_add": PreviewAdd,
    "preview_update": PreviewUpdate,
    "preview_delete": PreviewDelete,
    "preview_delete_all": PreviewDeleteAll,
    "add_note": AddNote,
    "update_note": UpdateNote,
    "delete_note": DeleteNote,
    "delete_all_notes": DeleteAllNotes,
}


def all_declarations() -> list[types.FunctionDeclaration]:
    """Every tool, in the format the API expects."""
    return [to_gemini_declaration(model, name) for name, model in TOOLS.items()]


def validate_call(name: str, args: dict):
    """Check one tool call from the LLM.

    Returns the validated Pydantic object, or raises ValidationError if the
    model sent something wrong (missing field, bad date, limit too big...).
    """
    model = TOOLS.get(name)
    if model is None:
        raise ValueError(f"Unknown tool: {name}")
    return model(**args)
