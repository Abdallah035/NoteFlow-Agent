"""Describe what a change would do, so the user can see it before agreeing."""


def build_diff(before: dict, changes: dict) -> list[str]:
    """Describe what would change, one line per field.

    Used in the confirmation message so the user knows exactly what they are
    saying yes to.
    """
    lines = []

    if changes.get("title") and changes["title"] != before["title"]:
        lines.append(f"title: {before['title']} -> {changes['title']}")

    if changes.get("body"):
        if changes.get("body_mode") == "replace":
            lines.append(f"body: replaced with '{changes['body']}'")
        else:
            lines.append(f"body: add '{changes['body']}'")

    if changes.get("tags") is not None:
        if changes.get("tags_mode") == "replace":
            lines.append(f"tags: {before['tags']} -> {changes['tags']}")
        else:
            new_tags = [t for t in changes["tags"] if t not in before["tags"]]
            if new_tags:
                lines.append(f"tags: add {new_tags}")

    if not lines:
        lines.append("(no actual change)")

    return lines
