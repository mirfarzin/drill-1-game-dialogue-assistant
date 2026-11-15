"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
from pathlib import Path
import pytest

from app.core import (
    END_TOKEN,
    START_TOKEN,
    DialogueCorpus,
    detokenize,
    milestone_1,
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement dialogue generation logic using the model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
