"""Chat with the agent in the terminal.

    python -m noteflow.cli
"""

import os

from dotenv import load_dotenv

from noteflow import db, embeddings
from noteflow.orchestrator import Agent

load_dotenv()


def main():
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("No API key. Put GOOGLE_API_KEY=... in your .env file.")
        return

    conn = db.connect("noteflow.db")
    db.init_schema(conn)
    agent = Agent(conn)

    # Load the model now so the first search is not slow.
    if embeddings.enabled():
        print("Loading the embedding model...")
        embeddings.embed("warm up")

    print("NoteFlow - talk to your notes. Empty line to quit.\n")

    while True:
        try:
            message = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not message:
            break

        answer = agent.handle(message)
        print(f"\nagent: {answer['reply']}")

        # Show the buttons as plain text in the terminal.
        for option in answer["options"]:
            print(f"   {option['label']}")
        print()

    print("Bye.")


if __name__ == "__main__":
    main()
