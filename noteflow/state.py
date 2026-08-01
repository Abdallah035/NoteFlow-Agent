"""What the agent remembers between messages in one conversation."""

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Memory for one chat session."""

    # The note we last talked about, so "that note" means something.
    last_note_id: int | None = None

    # The notes we last showed the user.
    last_results: list = field(default_factory=list)

    # A change we have shown the user and are waiting for them to accept.
    # Holds {"tool": name, "args": dict, "note": note}. The writing tools
    # refuse to run unless this matches, so nothing is saved without a yes.
    pending_action: dict | None = None

    # The conversation so far, sent back to the model for context.
    history: list = field(default_factory=list)

    def clear_pending(self):
        """Forget the change we were waiting on."""
        self.pending_action = None
