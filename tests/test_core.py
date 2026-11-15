"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import math
from pathlib import Path
import pytest

from app.core import (
    END_TOKEN,
    START_TOKEN,
    CharacterDialogueModelManager,
    DialogueCorpus,
    DialogueGenerator,
    NGramModel,
    detokenize,
    milestone_1,
    milestone_2,
    milestone_3,
    normalize_text,
    pad_sentence,
    tokenize,
)
from app.storage import Store


# Milestone 1: Collect and preprocess a corpus of game dialogues.
def test_milestone_1(tmp_path, monkeypatch):
    """Test milestone 1 preprocessing, corpus loading, and metadata persistence."""
    # Point storage to isolated tmp_path
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))

    corpus = milestone_1()
    assert isinstance(corpus, DialogueCorpus)
    assert len(corpus) >= 100

    # Verify 4 distinct NPC characters
    chars = corpus.get_characters()
    assert chars == [
        "elindra_mage",
        "garrick_blacksmith",
        "kaelen_rogue",
        "pip_bard",
    ]

    for cid in chars:
        sentences = corpus.get_sentences(cid)
        assert len(sentences) >= 25
        vocab = corpus.get_vocabulary(cid)
        assert len(vocab) > 50

    # Check store persistence
    store = Store()
    meta = store.get("milestone_1")
    assert meta is not None
    assert meta["status"] == "completed"
    assert meta["total_dialogues"] == len(corpus)
    assert set(meta["characters"]) == set(chars)


def test_preprocessing_normalization_and_idempotency():
    """Verify Unicode NFKC normalization and idempotency."""
    raw = "  Cold　iron   bends  for no one... aren't we sure?  "
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice
    assert once == "cold iron bends for no one... aren't we sure?"


def test_tokenization_and_padding():
    """Verify punctuation isolation and boundary marker padding."""
    line = "Keep your hands off the anvil, lad!"
    tokens = tokenize(line)
    assert tokens == ["keep", "your", "hands", "off", "the", "anvil", ",", "lad", "!"]

    padded = pad_sentence(tokens, n=3)
    assert padded == [START_TOKEN, START_TOKEN] + tokens + [END_TOKEN]


def test_detokenize_dialogue():
    """Verify detokenization spacing and capitalization."""
    tokens = [START_TOKEN, "a", "song", "in", "the", "tavern", "!", END_TOKEN]
    assert detokenize(tokens) == "A song in the tavern!"


# Milestone 2: Build the n-gram language model with smoothing.
def test_milestone_2(tmp_path, monkeypatch):
    """Test milestone 2 language model building, character isolation, and metadata."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))

    corpus = milestone_1()
    manager = milestone_2(corpus=corpus, n=3, k=1.0)
    assert isinstance(manager, CharacterDialogueModelManager)

    # Check store metadata
    store = Store()
    meta = store.get("milestone_2")
    assert meta is not None
    assert meta["status"] == "completed"
    assert meta["ngram_order"] == 3
    assert meta["smoothing_k"] == 1.0


def test_ngram_counting_bigram_and_trigram():
    """Verify raw frequency counts for bigrams and trigrams."""
    model = NGramModel(n=3)
    sentences = [
        ["the", "fire", "burns", "bright"],
        ["the", "fire", "needs", "coal"],
    ]
    model.train(sentences)

    # Bigram counts: ('the',) -> 'fire' is 2
    assert model.count(("the",), "fire") == 2
    # Trigram counts: ('the', 'fire') -> 'burns' is 1, 'needs' is 1
    assert model.count(("the", "fire"), "burns") == 1
    assert model.count(("the", "fire"), "needs") == 1
    assert model.count(("the", "fire"), "water") == 0
    assert model.context_count(("the", "fire")) == 2


def test_mle_and_laplace_smoothing_arithmetic():
    """Verify MLE and Add-k Laplace smoothing calculation formulas."""
    model = NGramModel(n=2)
    sentences = [["cold", "iron"]]
    model.train(sentences)
    # Vocab: 'cold', 'iron', '</s>' => size 3
    v_size = model.vocab_size
    assert v_size == 3

    # MLE (k=0.0): P('iron' | 'cold') = 1 / 1 = 1.0
    p_mle = model.prob("iron", ("cold",), k=0.0)
    assert p_mle == 1.0

    # Add-1 Laplace (k=1.0): (1 + 1) / (1 + 1 * 3) = 2 / 4 = 0.5
    p_laplace = model.prob("iron", ("cold",), k=1.0)
    assert p_laplace == 0.5

    # Unseen in context with k=1.0: P('cold' | 'cold') = (0 + 1) / (1 + 3) = 0.25
    p_unseen_trans = model.prob("cold", ("cold",), k=1.0)
    assert p_unseen_trans == 0.25

    # Completely unseen context with k=1.0: P('iron' | 'hot') = (0 + 1) / (0 + 3) = 1/3
    p_unseen_ctx = model.prob("iron", ("hot",), k=1.0)
    assert math.isclose(p_unseen_ctx, 1.0 / 3.0, rel_tol=1e-5)


def test_probability_distribution_normalization():
    """Verify sum of probabilities across the vocabulary equals 1.0."""
    model = NGramModel(n=3)
    sentences = [
        ["good", "steel", "does", "not", "whine"],
        ["a", "dull", "sword", "is", "worse"],
    ]
    model.train(sentences)

    context = ("good", "steel")
    dist = model.get_distribution(context, k=1.0)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-5)

    # Completely unseen context with pure Laplace (use_backoff=False)
    unseen_context = ("ancient", "parchment")
    unseen_dist = model.get_distribution(unseen_context, k=1.0, use_backoff=False)
    assert math.isclose(sum(unseen_dist.values()), 1.0, rel_tol=1e-5)
    # Under pure Laplace, unseen context gives uniform 1 / |V|
    expected_uniform = 1.0 / model.vocab_size
    for p in unseen_dist.values():
        assert math.isclose(p, expected_uniform, rel_tol=1e-5)

    # Unseen context with backoff (use_backoff=True)
    backoff_dist = model.get_distribution(unseen_context, k=1.0, use_backoff=True)
    assert math.isclose(sum(backoff_dist.values()), 1.0, rel_tol=1e-5)


def test_character_specific_model_isolation():
    """Verify that NPC characters maintain isolated vocabulary distributions."""
    corpus = milestone_1()
    manager = CharacterDialogueModelManager(n=3, k=1.0)
    manager.train_from_corpus(corpus)

    garrick_model = manager.get_model("garrick_blacksmith")
    elindra_model = manager.get_model("elindra_mage")
    kaelen_model = manager.get_model("kaelen_rogue")
    pip_model = manager.get_model("pip_bard")

    # Garrick has anvil/hammer terms, Elindra/Pip/Kaelen do not
    assert garrick_model.unigram_counts["anvil"] > 0
    assert elindra_model.unigram_counts["anvil"] == 0
    assert kaelen_model.unigram_counts["anvil"] == 0
    assert pip_model.unigram_counts["anvil"] == 0

    # Elindra has astral/runes terms, Garrick does not
    assert elindra_model.unigram_counts["runes"] > 0
    assert garrick_model.unigram_counts["runes"] == 0

    # Pip has lute/ballad terms, Garrick does not
    assert pip_model.unigram_counts["lute"] > 0
    assert garrick_model.unigram_counts["lute"] == 0


# Milestone 3: Implement dialogue generation logic using the model.
def test_milestone_3(tmp_path, monkeypatch):
    """Test milestone 3 dialogue generation and metadata persistence."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))

    output = milestone_3(character_id="garrick_blacksmith", prompt="Good steel", seed=42)
    assert isinstance(output, str)
    assert output.lower().startswith("good steel")

    store = Store()
    meta = store.get("milestone_3")
    assert meta is not None
    assert meta["status"] == "completed"
    assert meta["character"] == "garrick_blacksmith"


def test_generation_empty_prompt_and_reproducibility():
    """Verify empty prompt priming and reproducible generation with seed."""
    corpus = milestone_1()
    manager = CharacterDialogueModelManager(n=3, k=0.05)
    manager.train_from_corpus(corpus)

    model = manager.get_model("pip_bard")
    gen1 = DialogueGenerator(model=model, seed=123)
    gen2 = DialogueGenerator(model=model, seed=123)

    out1 = gen1.generate(prompt="", max_tokens=15)
    out2 = gen2.generate(prompt="", max_tokens=15)

    assert len(out1) > 0
    assert out1 == out2


def test_generation_max_tokens_and_alternatives():
    """Verify max token limitation and multi-candidate generation."""
    corpus = milestone_1()
    manager = CharacterDialogueModelManager(n=3, k=0.05)
    manager.train_from_corpus(corpus)

    model = manager.get_model("elindra_mage")
    gen = DialogueGenerator(model=model, seed=99)

    alts = gen.generate_alternatives(prompt="The stars", max_tokens=10, count=3)
    assert len(alts) == 3
    for alt in alts:
        assert alt.lower().startswith("the stars")
        assert len(alt.split()) <= 15


def test_generation_character_consistency():
    """Verify character vocabulary style during generation."""
    corpus = milestone_1()
    manager = CharacterDialogueModelManager(n=3, k=0.0)
    manager.train_from_corpus(corpus)

    garrick_gen = DialogueGenerator(model=manager.get_model("garrick_blacksmith"), seed=42)
    elindra_gen = DialogueGenerator(model=manager.get_model("elindra_mage"), seed=42)

    garrick_text = garrick_gen.generate(prompt="Good steel", max_tokens=20)
    elindra_text = elindra_gen.generate(prompt="The stars", max_tokens=20)

    assert any(w in garrick_text for w in ["iron", "shield", "coal", "pommel", "blade", "anvil", "tools", "whine", "forge"])
    assert any(w in elindra_text for w in ["stars", "align", "geometries", "spire", "runes", "ether", "veil", "prophecy"])


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
