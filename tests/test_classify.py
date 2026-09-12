"""Guards the compact classifier prompt: rendering the real docs/taxonomy.md must fail loudly when an
intent loses what the block needs. Needs no Ollama, so it runs in CI."""

import pytest

from src import classify

TAXONOMY = classify.TAXONOMY_MD.read_text(encoding="utf-8")
INTENTS, _ = classify.parse_taxonomy(TAXONOMY)


def section_bounds(lines: list[str], intent: str) -> tuple[int, int]:
    """Line index of the intent's heading and of the next heading (or the end of the file)."""
    start = lines.index(f"## {intent}")
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return start, end


def without_summary(intent: str) -> str:
    """The real taxonomy with this intent's Summary line deleted."""
    lines = TAXONOMY.splitlines()
    start, end = section_bounds(lines, intent)
    kept = []
    for index, line in enumerate(lines):
        if start < index < end and line.startswith(classify.SUMMARY_PREFIX):
            continue
        kept.append(line)
    return "\n".join(kept)


def with_one_example(intent: str) -> str:
    """The real taxonomy with every example of this intent but the first deleted (near misses untouched)."""
    lines = TAXONOMY.splitlines()
    start, end = section_bounds(lines, intent)
    position = lines.index("**Examples**", start, end) + 1
    dropped = []
    seen = 0
    while not lines[position].startswith("**"):  # the Examples list ends at the Near misses heading
        if lines[position].startswith("- [case "):
            seen += 1
            if seen > 1:
                dropped.append(position)
        position += 1
    return "\n".join(line for index, line in enumerate(lines) if index not in dropped)


def test_real_taxonomy_renders_every_intent_with_two_short_examples() -> None:
    block = classify.render_block(TAXONOMY)
    assert len(INTENTS) == 9
    for intent in INTENTS:
        assert f"\n{intent}: " in block
    example_lines = [line for line in block.splitlines() if line.startswith('- "')]
    assert len(example_lines) == 2 * len(INTENTS)
    for line in example_lines:
        assert len(line) <= len('- ""') + classify.MAX_EXAMPLE_CHARS


@pytest.mark.parametrize("intent", INTENTS)
def test_missing_summary_line_raises(intent: str) -> None:
    broken = without_summary(intent)
    assert len(broken.splitlines()) == len(TAXONOMY.splitlines()) - 1  # the edit really removed a line
    with pytest.raises(ValueError, match=intent):
        classify.render_block(broken)


@pytest.mark.parametrize("intent", INTENTS)
def test_fewer_than_two_examples_raises(intent: str) -> None:
    broken = with_one_example(intent)
    assert len(broken.splitlines()) < len(TAXONOMY.splitlines())  # the edit really removed examples
    with pytest.raises(ValueError, match=intent):
        classify.render_block(broken)
