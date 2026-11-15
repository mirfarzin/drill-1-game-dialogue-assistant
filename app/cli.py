"""Command-line entry point: `python -m app.cli --help`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from app.core import (
    CharacterDialogueModelManager,
    DialogueCorpus,
    DialogueGenerator,
    milestone_1,
    milestone_4,
)


def format_character_list(corpus: DialogueCorpus) -> str:
    """Format available characters into a readable table."""
    lines = ["Available Characters:"]
    for cid in corpus.get_characters():
        name = corpus.get_character_name(cid)
        count = len(corpus.get_sentences(cid))
        vocab = len(corpus.get_vocabulary(cid))
        lines.append(f"  • {cid:<20} | {name:<30} ({count} lines, {vocab} words)")
    return "\n".join(lines)


def run_interactive(corpus: DialogueCorpus, manager: CharacterDialogueModelManager, k: float, length: int) -> None:
    """Run an interactive dialogue studio session."""
    characters = corpus.get_characters()
    print("\n=======================================================")
    print(" 🎮 Game Dialogue Assistant — Interactive Studio")
    print("=======================================================\n")
    print(format_character_list(corpus))
    print("\nCommands: 'exit' or 'quit' to quit, 'chars' to list characters, ':char <id>' to switch.\n")

    current_char = characters[0]
    print(f"Active Character: {corpus.get_character_name(current_char)} [{current_char}]")

    while True:
        try:
            line = input(f"\n[{current_char}] Enter prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Farewell!")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("Farewell, adventurer!")
            break
        if line.lower() == "chars":
            print(format_character_list(corpus))
            continue
        if line.startswith(":char "):
            new_char = line.split(maxsplit=1)[1].strip()
            if new_char in characters:
                current_char = new_char
                print(f"Switched to: {corpus.get_character_name(current_char)}")
            else:
                print(f"Unknown character '{new_char}'. Available: {', '.join(characters)}")
            continue

        model = manager.get_model(current_char)
        gen = DialogueGenerator(model=model, fallback_model=manager.global_model, k=k)
        generated = gen.generate(prompt=line, max_tokens=length)
        cname = corpus.get_character_name(current_char)
        print(f"\n🗣️  {cname}:\n   \"{generated}\"")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="game-dialogue-assistant",
        description="Game designers use this tool to generate dialogue snippets for NPCs based on a corpus of game scripts, ensuring character consistency and rapid prototyping.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. hello (from starter plumbing)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. characters
    sub.add_parser("characters", help="list available NPC characters in the dialogue corpus")

    # 3. generate
    gen_p = sub.add_parser("generate", help="generate dialogue snippet from a starting phrase")
    gen_p.add_argument("-c", "--character", default="garrick_blacksmith", help="NPC character ID")
    gen_p.add_argument("-p", "--prompt", default="", help="Starting phrase or sentence seed")
    gen_p.add_argument("-l", "--length", type=int, default=25, help="Maximum generated tokens")
    gen_p.add_argument("-n", "--ngram", type=int, default=3, choices=[2, 3, 4], help="N-gram order (2 or 3)")
    gen_p.add_argument("-k", "--smoothing", type=float, default=0.05, help="Laplace add-k parameter")
    gen_p.add_argument("-a", "--alternatives", type=int, default=1, help="Number of alternatives to generate")
    gen_p.add_argument("-s", "--seed", type=int, default=None, help="Optional random seed")
    gen_p.add_argument("-o", "--output-file", default=None, help="Save generated dialogue to a text file")

    # 4. interactive
    int_p = sub.add_parser("interactive", help="launch interactive terminal studio session")
    int_p.add_argument("-k", "--smoothing", type=float, default=0.05, help="Laplace smoothing parameter")
    int_p.add_argument("-l", "--length", type=int, default=25, help="Maximum generated tokens")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Game Dialogue Assistant is ready. Open README.md for your milestones.")
        return 0

    corpus = milestone_1()

    if args.command == "characters":
        print(format_character_list(corpus))
        return 0

    if args.command == "interactive":
        manager = CharacterDialogueModelManager(n=3, k=args.smoothing)
        manager.train_from_corpus(corpus)
        run_interactive(corpus, manager, k=args.smoothing, length=args.length)
        return 0

    if args.command == "generate":
        available_chars = corpus.get_characters()
        if args.character not in available_chars:
            print(f"Error: Unknown character '{args.character}'. Available: {', '.join(available_chars)}", file=sys.stderr)
            return 1

        outputs = milestone_4(
            character_id=args.character,
            prompt=args.prompt,
            length=args.length,
            alternatives=args.alternatives,
            ngram=args.ngram,
            k=args.smoothing,
            seed=args.seed,
        )

        char_name = corpus.get_character_name(args.character)
        output_lines = [
            f"Character: {char_name} [{args.character}]",
            f"N-gram Order: {args.ngram} | Laplace Smoothing k: {args.smoothing}",
        ]
        if args.prompt:
            output_lines.append(f"Seed Prompt: \"{args.prompt}\"")
        output_lines.append("Generated Dialogue:")

        for i, snippet in enumerate(outputs, 1):
            if len(outputs) > 1:
                output_lines.append(f"  [{i}] \"{snippet}\"")
            else:
                output_lines.append(f"  \"{snippet}\"")

        rendered = "\n".join(output_lines)
        print(rendered)

        if args.output_file:
            try:
                out_path = Path(args.output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(rendered + "\n", encoding="utf-8")
                print(f"\n[Saved to {out_path}]")
            except Exception as e:
                print(f"Error saving to file: {e}", file=sys.stderr)
                return 1

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
