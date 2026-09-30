"""
Classify messages with the trained model.

  python predict.py                       # runs built-in examples, then interactive mode
  python predict.py "Win a free prize!"   # classify a single message
"""

import sys

from spam_classifier import load_model, predict

EXAMPLES = [
    "Congratulations! You've won a $1000 gift card. Click http://bit.ly/claim now to claim your prize!",
    "URGENT! Your account has been suspended. Reply with your PIN to verify immediately.",
    "Hey, are we still meeting for lunch tomorrow at 1?",
    "Can you send me the project report before the review meeting?",
    "FREE entry into our weekly draw! Text WIN to 80082 to receive your reward.",
    "Thanks for the notes, I'll go through them tonight and get back to you.",
]


def main():
    model = load_model()

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(f"{predict(model, message).upper()}: {message}")
        return

    print("Built-in examples\n" + "-" * 60)
    for msg in EXAMPLES:
        print(f"[{predict(model, msg).upper():4}] {msg}")

    print("\nInteractive mode - type a message (or 'q' to quit)")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"q", "quit", "exit"}:
            break
        if msg:
            print(f"   -> {predict(model, msg).upper()}")


if __name__ == "__main__":
    main()
