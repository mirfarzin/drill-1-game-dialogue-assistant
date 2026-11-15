# Game Dialogue Assistant

This Game Dialogue Assistant is a tool designed for game designers. It helps generate dialogue snippets for Non-Player Characters (NPCs) by leveraging a next-word predictor model trained on a corpus of game scripts. The goal is to ensure character consistency and facilitate rapid prototyping of game narratives.

## What you are building

**Goal.** You will build a command-line tool that helps game designers generate dialogue snippets for NPCs, ensuring character consistency and rapid prototyping.

**Done when.** Your tool can collect and preprocess game dialogue, build an n-gram language model with smoothing, and generate coherent dialogue snippets from a starting phrase via a command-line interface. The text generation capability will produce consistent dialogue based on learned patterns.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Use text processing techniques to clean and prepare your dialogue data.
- [ ] **2. Build the n-gram language model with smoothing.**  
      Hint: Implement the N-gram model and apply smoothing techniques like Laplace smoothing.
- [ ] **3. Implement dialogue generation logic using the model.**  
      Hint: Develop the logic to iteratively predict the next word based on the model.
- [ ] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Create a simple command-line interface to take user input and display generated text.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the desired length of the generated dialogue snippet.
- Implement an option to generate multiple alternative dialogue snippets for a given starting phrase.
- Add a feature to save the generated dialogue to a text file.
