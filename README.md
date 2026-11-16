[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/zdqgtzrexp/badge.svg)](https://ask.thili.ai/b/zdqgtzrexp)

# Game Dialogue Assistant

This Game Dialogue Assistant is a tool designed for game designers. It helps generate dialogue snippets for Non-Player Characters (NPCs) by leveraging a next-word predictor model trained on a corpus of game scripts. The goal is to ensure character consistency and facilitate rapid prototyping of game narratives.

## What you are building

**Goal.** You will build a command-line tool that helps game designers generate dialogue snippets for NPCs, ensuring character consistency and rapid prototyping.

**Done when.** Your tool can collect and preprocess game dialogue, build an n-gram language model with smoothing, and generate coherent dialogue snippets from a starting phrase via a command-line interface. The text generation capability will produce consistent dialogue based on learned patterns.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Use text processing techniques to clean and prepare your dialogue data.
- [x] **2. Build the n-gram language model with smoothing.**  
      Hint: Implement the N-gram model and apply smoothing techniques like Laplace smoothing.
- [x] **3. Implement dialogue generation logic using the model.**  
      Hint: Develop the logic to iteratively predict the next word based on the model.
- [x] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Create a simple command-line interface to take user input and display generated text.

## N-Gram Language Model & Smoothing Details

### How Bigrams and Trigrams Work
An n-gram is a contiguous sequence of $n$ items from a given text:
- **Bigram ($N=2$)**: Pairs of consecutive words $(w_{i-1}, w_i)$ where the next word $w_i$ is predicted conditioned on the single preceding word $w_{i-1}$.
- **Trigram ($N=3$)**: Triples of consecutive words $(w_{i-2}, w_{i-1}, w_i)$ where the next word $w_i$ is predicted conditioned on the two preceding words $(w_{i-2}, w_{i-1})$.
- **Boundary Markers**: Sentences are padded with start tokens (`<s>`) and end tokens (`</s>`) so the model learns natural dialogue openings and terminations.

### Maximum Likelihood Estimation (MLE)
The unsmoothed probability of observing token $w$ given context history $h = (w_1, \dots, w_{n-1})$ is computed directly from dialogue co-occurrence counts:
$$P_{\text{MLE}}(w \mid h) = \frac{\text{Count}(h, w)}{\text{Count}(h)}$$

### Add-$k$ Laplace Smoothing
To resolve the zero-frequency problem where unseen word combinations would receive zero probability, Add-$k$ Laplace smoothing redistributes probability mass uniformly across the entire active vocabulary $V$:
$$P_{\text{Laplace}}(w \mid h) = \frac{\text{Count}(h, w) + k}{\text{Count}(h) + k \cdot |V|}$$

- For $k = 1.0$, this computes standard **Laplace smoothing**.
- For $0 < k < 1.0$, it applies **Lidstone smoothing**, preventing high pseudo-counts from overwhelming observed frequencies on compact dialogue datasets.
- For $k = 0.0$, it falls back to exact MLE.
- For every context $h$, the probabilities strictly sum to $1.0$: $\sum_{w \in V} P(w \mid h) = 1.0$.

### Character-Conditioned Modeling
To ensure NPC personality consistency, the model manager maintains isolated models per character:
- `garrick_blacksmith`: Trained exclusively on blacksmith lines (anvil, forge, hammer, iron).
- `elindra_mage`: Trained exclusively on mage lines (stars, ether, runes, veil, astral).
- `pip_bard`: Trained exclusively on bard lines (ballad, cider, lute, tavern, merriment).
- `kaelen_rogue`: Trained exclusively on rogue lines (dagger, coin, shadows, alleyways).

Each character model possesses its own isolated vocabulary, preventing one character's speaking style from mixing into another's.

### Test Verification
- All **21 unit & smoke tests** pass (`tests/test_core.py` and `tests/test_smoke.py`).
- 0 skipped, 0 failed.

## CLI Usage Guide

```bash
# Check plumbing command
python -m app.cli hello

# List all available NPC characters and line counts
python -m app.cli characters

# Generate dialogue for Garrick the Blacksmith
python -m app.cli generate -c garrick_blacksmith -p "Good steel"

# Generate dialogue for Elindra the Mage with custom length
python -m app.cli generate -c elindra_mage -p "The stars" -l 20

# Generate multiple alternatives and save to a text file
python -m app.cli generate -c pip_bard -p "A song" -a 3 -o generated_ballads.txt

# Launch interactive terminal studio session
python -m app.cli interactive
```

### CLI Command Options

| Command / Option | Description | Default |
|---|---|---|
| `hello` | Verify baseline starter plumbing | N/A |
| `characters` | List available NPC character IDs and stats | N/A |
| `generate` | Generate dialogue snippet from a starting phrase | N/A |
| `-c, --character` | NPC ID (`garrick_blacksmith`, `elindra_mage`, `pip_bard`, `kaelen_rogue`) | `garrick_blacksmith` |
| `-p, --prompt` | Starting phrase or sentence seed | `""` |
| `-l, --length` | Maximum generated tokens | `25` |
| `-n, --ngram` | N-gram order (2 or 3) | `3` |
| `-k, --smoothing` | Laplace add-$k$ parameter (0 for MLE, 0.05 for balanced) | `0.05` |
| `-a, --alternatives`| Number of alternative snippets to generate | `1` |
| `-s, --seed` | Random seed for reproducible generation | `None` |
| `-o, --output-file`| Path to save generated dialogue | `None` |
| `interactive` | Launch real-time interactive terminal studio session | N/A |

### Example Outputs

#### Garrick the Grumpy Blacksmith
```bash
python -m app.cli generate -c garrick_blacksmith -p "Good steel" -k 0.0
```
```
Character: Garrick the Grumpy Blacksmith [garrick_blacksmith]
N-gram Order: 3 | Laplace Smoothing k: 0.0
Seed Prompt: "Good steel"
Generated Dialogue:
  "Good steel does not bargain with fools who break tools."
```

#### Elindra the Cryptic Mage
```bash
python -m app.cli generate -c elindra_mage -p "The stars" -k 0.0
```
```
Character: Elindra the Cryptic Mage [elindra_mage]
N-gram Order: 3 | Laplace Smoothing k: 0.0
Seed Prompt: "The stars"
Generated Dialogue:
  "The stars align in fractured geometries above the weeping spire."
```

#### Pip the Cheerful Bard
```bash
python -m app.cli generate -c pip_bard -p "A song" -k 0.0
```
```
Character: Pip the Cheerful Bard [pip_bard]
N-gram Order: 3 | Laplace Smoothing k: 0.0
Seed Prompt: "A song"
Generated Dialogue:
  "A song in the tavern fire is warm and the ballad of the golden feather!"
```

#### Kaelen the Shadow Rogue
```bash
python -m app.cli generate -c kaelen_rogue -p "Shadows are" -k 0.0
```
```
Character: Kaelen the Shadow Rogue [kaelen_rogue]
N-gram Order: 3 | Laplace Smoothing k: 0.0
Seed Prompt: "Shadows are"
Generated Dialogue:
  "Shadows are kinder allies than honest promises in this city."
```

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
