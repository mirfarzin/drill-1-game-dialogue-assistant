"""Your code. One stub per milestone; build them in order and keep each one small."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple
import unicodedata

from app.storage import Store


START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "dialogues.json"

# Regex pattern matching words (with internal apostrophes) or punctuation
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[.,!?;:]")


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize text using Unicode NFKC composition and whitespace cleaning.

    Ensures idempotency: normalize_text(normalize_text(x)) == normalize_text(x)
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("’", "'").replace("`", "'")
    if lowercase:
        normalized = normalized.lower()
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize dialogue into words and distinct punctuation tokens."""
    cleaned = normalize_text(text, lowercase=lowercase)
    if not cleaned:
        return []
    return TOKEN_PATTERN.findall(cleaned)


def pad_sentence(
    tokens: List[str],
    n: int = 3,
    start_token: str = START_TOKEN,
    end_token: str = END_TOKEN,
) -> List[str]:
    """Pad tokens with (n - 1) start markers and 1 end marker."""
    prefix = [start_token] * max(1, n - 1)
    return prefix + list(tokens) + [end_token]


def detokenize(tokens: List[str], remove_special: bool = True) -> str:
    """Reconstruct natural formatted dialogue from a token sequence."""
    special = {START_TOKEN, END_TOKEN, UNK_TOKEN}
    filtered = [t for t in tokens if not (remove_special and t in special)]
    if not filtered:
        return ""

    result: List[str] = []
    no_space_before = {".", ",", "!", "?", ";", ":"}
    for i, token in enumerate(filtered):
        if i > 0 and token not in no_space_before:
            result.append(" ")
        result.append(token)

    sentence = "".join(result).strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


class DialogueCorpus:
    """In-memory representation of the preprocessed multi-character game dialogue corpus."""

    def __init__(self, raw_items: Optional[List[dict]] = None, lowercase: bool = True) -> None:
        self.lowercase = lowercase
        self.entries: List[dict] = []
        self._by_character: Dict[str, List[List[str]]] = {}
        self._character_names: Dict[str, str] = {}

        if raw_items:
            for item in raw_items:
                cid = item.get("character") or item.get("character_id", "unknown")
                cname = item.get("character_name", cid)
                text = item.get("text", "")
                tokens = tokenize(text, lowercase=self.lowercase)
                if tokens:
                    self.entries.append({
                        "character": cid,
                        "character_name": cname,
                        "text": text,
                        "tokens": tokens,
                    })
                    if cid not in self._by_character:
                        self._by_character[cid] = []
                        self._character_names[cid] = cname
                    self._by_character[cid].append(tokens)

    @classmethod
    def from_json(cls, file_path: str | Path, lowercase: bool = True) -> DialogueCorpus:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dialogue corpus not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(raw_items=data, lowercase=lowercase)

    def get_characters(self) -> List[str]:
        return sorted(self._by_character.keys())

    def get_character_name(self, character_id: str) -> str:
        return self._character_names.get(character_id, character_id)

    def get_sentences(self, character_id: Optional[str] = None) -> List[List[str]]:
        if character_id is None:
            return [e["tokens"] for e in self.entries]
        return self._by_character.get(character_id, [])

    def get_vocabulary(self, character_id: Optional[str] = None) -> Set[str]:
        vocab: Set[str] = set()
        for tokens in self.get_sentences(character_id):
            vocab.update(tokens)
        return vocab

    def __len__(self) -> int:
        return len(self.entries)


# --- Milestone 1: Collect and preprocess a corpus of game dialogues.
# Hint: Use text processing techniques to clean and prepare your dialogue data.
def milestone_1(data_path: str | Path | None = None) -> DialogueCorpus:  # TODO(milestone 1)
    """Load and preprocess the game dialogue corpus.

    Cleans dialogue lines, extracts tokens and sentence boundaries,
    and stores character distribution summaries in Store.
    """
    path = Path(data_path or DEFAULT_DATA_PATH)
    corpus = DialogueCorpus.from_json(path, lowercase=True)

    # Persist summary metrics to storage
    store = Store()
    characters = corpus.get_characters()
    store.put("milestone_1", {
        "status": "completed",
        "total_dialogues": len(corpus),
        "characters": characters,
        "character_counts": {cid: len(corpus.get_sentences(cid)) for cid in characters},
        "vocabulary_size": len(corpus.get_vocabulary()),
    })

    return corpus


class NGramModel:
    """N-gram language model supporting count-based estimation and Laplace smoothing."""

    def __init__(self, n: int = 3, character_id: Optional[str] = None) -> None:
        if n < 1:
            raise ValueError(f"N-gram order must be >= 1, got {n}")
        self.n: int = n
        self.character_id: Optional[str] = character_id

        # Hierarchical counts: order -> context_tuple -> Counter(target_word -> count)
        self.counts_by_order: Dict[int, Dict[Tuple[str, ...], Counter[str]]] = {
            order: defaultdict(Counter) for order in range(1, n + 1)
        }
        # Context occurrences: order -> context_tuple -> total occurrences
        self.context_counts_by_order: Dict[int, Counter[Tuple[str, ...]]] = {
            order: Counter() for order in range(1, n + 1)
        }
        self.vocab: Set[str] = set()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def ngram_counts(self) -> Dict[Tuple[str, ...], Counter[str]]:
        return self.counts_by_order[self.n]

    @property
    def context_counts(self) -> Counter[Tuple[str, ...]]:
        return self.context_counts_by_order[self.n]

    @property
    def unigram_counts(self) -> Counter[str]:
        return self.counts_by_order[1][()]

    def train(self, sentences: Sequence[Sequence[str]]) -> None:
        """Accumulate n-gram counts across orders 1..n from tokenized sentences."""
        for sentence in sentences:
            if not sentence:
                continue

            padded = pad_sentence(list(sentence), n=self.n)
            for token in padded:
                if token != START_TOKEN:
                    self.vocab.add(token)

            for order in range(1, self.n + 1):
                for i in range(len(padded) - order + 1):
                    sub = padded[i : i + order]
                    context = tuple(sub[:-1])
                    target = sub[-1]
                    if target == START_TOKEN:
                        continue
                    self.counts_by_order[order][context][target] += 1
                    self.context_counts_by_order[order][context] += 1

    def count(self, context: Tuple[str, ...], word: str) -> int:
        order = len(context) + 1
        if order in self.counts_by_order:
            return self.counts_by_order[order][context][word]
        return 0

    def context_count(self, context: Tuple[str, ...]) -> int:
        order = len(context) + 1
        if order in self.context_counts_by_order:
            return self.context_counts_by_order[order][context]
        return 0

    def prob(self, word: str, context: Tuple[str, ...], k: float = 1.0) -> float:
        """Compute conditional probability P(word | context).

        Uses Add-k (Laplace) smoothing:
            P(w | context) = (Count(context, w) + k) / (Count(context) + k * |V|)
        When k == 0 (MLE), returns 0.0 if Count(context) == 0.
        """
        c_ngram = self.count(context, word)
        c_context = self.context_count(context)
        v_size = max(1, self.vocab_size)

        if k <= 0.0:
            if c_context == 0:
                return 0.0
            return c_ngram / c_context

        return (c_ngram + k) / (c_context + (k * v_size))

    def get_distribution(
        self,
        context: Tuple[str, ...],
        k: float = 1.0,
        use_backoff: bool = False,
    ) -> Dict[str, float]:
        """Compute normalized probability distribution for a given context prefix."""
        candidates = sorted(self.vocab)
        v_size = max(1, len(candidates))
        c_context = self.context_count(context)

        if use_backoff and c_context == 0 and len(context) > 0:
            return self.get_distribution(context[1:], k=k, use_backoff=True)

        if k <= 0.0:
            if c_context == 0:
                uniform_p = 1.0 / v_size
                return {w: uniform_p for w in candidates}
            return {w: self.count(context, w) / c_context for w in candidates}

        denominator = c_context + (k * v_size)
        return {w: (self.count(context, w) + k) / denominator for w in candidates}


class CharacterDialogueModelManager:
    """Manages character-conditioned and global n-gram language models."""

    def __init__(self, n: int = 3, k: float = 1.0) -> None:
        self.n = n
        self.k = k
        self.models: Dict[str, NGramModel] = {}
        self.global_model: NGramModel = NGramModel(n=n, character_id="global")

    def train_from_corpus(self, corpus: DialogueCorpus) -> None:
        # Train global model on all sentences
        self.global_model.train(corpus.get_sentences())

        # Train isolated character models to preserve distinct style
        for char_id in corpus.get_characters():
            char_sentences = corpus.get_sentences(character_id=char_id)
            model = NGramModel(n=self.n, character_id=char_id)
            model.train(char_sentences)
            self.models[char_id] = model

    def get_model(self, character_id: Optional[str] = None) -> NGramModel:
        if character_id and character_id in self.models:
            return self.models[character_id]
        return self.global_model


# --- Milestone 2: Build the n-gram language model with smoothing.
# Hint: Implement the N-gram model and apply smoothing techniques like Laplace smoothing.
def milestone_2(
    corpus: Optional[DialogueCorpus] = None,
    n: int = 3,
    k: float = 1.0,
) -> CharacterDialogueModelManager:  # TODO(milestone 2)
    """Build and train the N-gram language models with Laplace smoothing.

    Args:
        corpus: Preprocessed DialogueCorpus instance (or None to load default).
        n: N-gram order (e.g. 2 for bigram, 3 for trigram).
        k: Add-k Laplace smoothing parameter.

    Returns:
        CharacterDialogueModelManager containing character-specific and global models.
    """
    if corpus is None:
        corpus = milestone_1()

    manager = CharacterDialogueModelManager(n=n, k=k)
    manager.train_from_corpus(corpus)

    # Persist milestone 2 metadata
    store = Store()
    characters = corpus.get_characters()
    store.put("milestone_2", {
        "status": "completed",
        "ngram_order": n,
        "smoothing_k": k,
        "character_vocab_sizes": {
            cid: manager.get_model(cid).vocab_size for cid in characters
        },
        "global_vocab_size": manager.global_model.vocab_size,
    })

    return manager


class DialogueGenerator:
    """Dialogue generation engine using n-gram language models with weighted sampling."""

    def __init__(
        self,
        model: NGramModel,
        fallback_model: Optional[NGramModel] = None,
        k: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.k = k
        self.rng = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    def _prepare_initial_context(
        self, prompt_tokens: List[str]
    ) -> Tuple[List[str], Tuple[str, ...]]:
        context_len = max(1, self.model.n - 1)
        if not prompt_tokens:
            return [], tuple([START_TOKEN] * context_len)

        output_tokens = list(prompt_tokens)
        if len(prompt_tokens) < context_len:
            pad_needed = context_len - len(prompt_tokens)
            context = tuple([START_TOKEN] * pad_needed + prompt_tokens)
        else:
            context = tuple(prompt_tokens[-context_len:])
        return output_tokens, context

    def _sample_next_token(self, context: Tuple[str, ...]) -> str:
        distribution = self.model.get_distribution(
            context, k=self.k, use_backoff=True
        )
        words = list(distribution.keys())
        probs = list(distribution.values())
        return self.rng.choices(words, weights=probs, k=1)[0]

    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 25,
        stop_at_end_token: bool = True,
    ) -> str:
        prompt_tokens = tokenize(prompt)
        output_tokens, context = self._prepare_initial_context(prompt_tokens)
        generated_count = 0

        while generated_count < max_tokens:
            next_token = self._sample_next_token(context)
            generated_count += 1

            if next_token == END_TOKEN:
                if stop_at_end_token:
                    if not output_tokens or output_tokens[-1] not in {".", "!", "?"}:
                        output_tokens.append(".")
                    break
                else:
                    continue

            output_tokens.append(next_token)
            context = tuple(list(context[1:]) + [next_token])

        return detokenize(output_tokens)

    def generate_alternatives(
        self,
        prompt: str = "",
        max_tokens: int = 25,
        count: int = 3,
    ) -> List[str]:
        return [self.generate(prompt=prompt, max_tokens=max_tokens) for _ in range(max(1, count))]


# --- Milestone 3: Implement dialogue generation logic using the model.
# Hint: Develop the logic to iteratively predict the next word based on the model.
def milestone_3(
    manager: Optional[CharacterDialogueModelManager] = None,
    character_id: str = "garrick_blacksmith",
    prompt: str = "Good steel",
    max_tokens: int = 25,
    k: float = 0.05,
    seed: Optional[int] = 42,
) -> str:  # TODO(milestone 3)
    """Generate dialogue using character-conditioned n-gram model.

    Args:
        manager: Trained CharacterDialogueModelManager (or None to train default).
        character_id: Target NPC character ID.
        prompt: Starting phrase or prompt.
        max_tokens: Maximum tokens to generate.
        k: Smoothing parameter.
        seed: Random seed for reproducible generation.

    Returns:
        Generated dialogue snippet.
    """
    if manager is None:
        manager = milestone_2()

    model = manager.get_model(character_id)
    generator = DialogueGenerator(
        model=model,
        fallback_model=manager.global_model,
        k=k,
        seed=seed,
    )
    snippet = generator.generate(prompt=prompt, max_tokens=max_tokens)

    # Persist milestone 3 metadata
    store = Store()
    store.put("milestone_3", {
        "status": "completed",
        "character": character_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "output": snippet,
    })

    return snippet


# --- Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
# Hint: Create a simple command-line interface to take user input and display generated text.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
