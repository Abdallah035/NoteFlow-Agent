"""Turn Pydantic models into tool declarations the LLM understands.

Pydantic writes JSON Schema in its own style. The Gemini API accepts only a
subset of it, so we strip the keys it rejects and flatten optional fields.
"""

from google.genai import types

from noteflow.schemas import AddNote, DeleteNote, SearchNotes, UpdateNote

# Keys Gemini's schema format does not accept.
_DROP = {
    "title",
    "default",
    "$defs",
    "additionalProperties",
    "$ref",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "anyOf",
}


def _clean(schema: dict) -> dict:
    """Remove keys the API rejects, keeping type, description, enum and items."""
    out = {}
    for key, value in schema.items():
        if key in _DROP:
            continue
        if key == "properties":
            out[key] = {name: _clean(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _clean(value)
        else:
            out[key] = value
    return out


def _unwrap_optional(field_schema: dict) -> dict:
    """Turn Pydantic's anyOf[X, null] for optional fields into plain X.

    'query: str | None' becomes anyOf[string, null], which Gemini rejects.
    The field stays optional by simply not being listed in "required".
    """
    if "anyOf" in field_schema:
        for option in field_schema["anyOf"]:
            if option.get("type") != "null":
                merged = dict(option)
                if "description" in field_schema:
                    merged["description"] = field_schema["description"]
                return merged
    return field_schema


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
TOOLS = {
    "add_note": AddNote,
    "search_notes": SearchNotes,
    "update_note": UpdateNote,
    "delete_note": DeleteNote,
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
