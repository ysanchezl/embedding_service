import pytest

from app.services.chunking import chunk_text


def count_words(units: list[str]) -> list[int]:
    """Fake tokenizer: one token per whitespace-separated word."""
    return [len(unit.split()) for unit in units]


def words(chunk: str) -> list[str]:
    return chunk.split()


def test_short_text_is_single_chunk():
    assert chunk_text("hola mundo", count_words, max_tokens=10, overlap_tokens=2) == ["hola mundo"]


def test_empty_or_blank_text_returns_no_chunks():
    assert chunk_text("", count_words, max_tokens=10, overlap_tokens=2) == []
    assert chunk_text("  \n\n ", count_words, max_tokens=10, overlap_tokens=2) == []


def test_chunks_never_exceed_max_tokens():
    text = " ".join(f"w{i}" for i in range(1000))

    chunks = chunk_text(text, count_words, max_tokens=50, overlap_tokens=5)

    assert len(chunks) > 1
    assert all(len(words(c)) <= 50 for c in chunks)


def test_no_text_is_lost():
    sentences = [f"Frase numero {i} del documento." for i in range(200)]
    text = " ".join(sentences)

    chunks = chunk_text(text, count_words, max_tokens=30, overlap_tokens=6)

    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_splits_on_sentence_boundaries_when_possible():
    text = "Primera frase corta. Segunda frase corta. Tercera frase corta. Cuarta frase corta."

    chunks = chunk_text(text, count_words, max_tokens=7, overlap_tokens=0)

    assert chunks == [
        "Primera frase corta. Segunda frase corta.",
        "Tercera frase corta. Cuarta frase corta.",
    ]


def test_respects_paragraphs_before_sentences():
    text = "uno dos tres\n\ncuatro cinco seis"

    chunks = chunk_text(text, count_words, max_tokens=4, overlap_tokens=0)

    assert chunks == ["uno dos tres", "cuatro cinco seis"]


def test_consecutive_chunks_overlap():
    text = " ".join(f"Frase {i} aqui." for i in range(40))

    chunks = chunk_text(text, count_words, max_tokens=12, overlap_tokens=3)

    for previous, current in zip(chunks, chunks[1:]):
        assert words(previous)[-3:] == words(current)[:3]


def test_overlap_larger_than_chunk_is_ignored_safely():
    text = " ".join(f"w{i}" for i in range(100))

    chunks = chunk_text(text, count_words, max_tokens=10, overlap_tokens=50)

    assert all(len(words(c)) <= 10 for c in chunks)
    assert len(chunks) < 100


def test_single_oversized_word_is_hard_split():
    def count_chars_per_4(units: list[str]) -> list[int]:
        return [max(1, len(u) // 4) if u.strip() else 0 for u in units]

    text = "x" * 400  # no spaces, 100 "tokens"

    chunks = chunk_text(text, count_chars_per_4, max_tokens=20, overlap_tokens=0)

    assert len(chunks) >= 5
    assert "".join(chunks) == text
    assert all(count_chars_per_4([c])[0] <= 20 for c in chunks)


def test_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        chunk_text("hola", count_words, max_tokens=0, overlap_tokens=0)
