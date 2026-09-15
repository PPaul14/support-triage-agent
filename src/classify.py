"""Stage 4 (classify), part 1: render the compact classifier prompt block from docs/taxonomy.md.

The guideline stays the single source of truth. The block is a smaller view of it, rebuilt from
each intent's Summary line, its first examples and the tie-break order.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from src import llm

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_MD = REPO_ROOT / "docs" / "taxonomy.md"
PROMPT_TXT = REPO_ROOT / "artifacts" / "classifier_prompt.txt"
CLASSIFY_MODEL = "phi3:3.8b-mini-128k-instruct-q4_0"
TIE_BREAK_SECTION = "Tie-break rules"
SUMMARY_PREFIX = "**Summary.** "
EXAMPLES_PER_INTENT = 2
MAX_EXAMPLE_CHARS = 120
EXAMPLE_LINE = re.compile(r'^- \[case \d+\] "(.*?)"(?=$| →| \()')  # the quote ends where a note or label starts
ORDER_LINE = re.compile(r"^\d+\. (\w+)$")
# The intent question asked after the block. golden.py's estimates use it, and src/model_label.py turns those
# cached estimates into the model labels.
ESTIMATE_INSTRUCTION = ("\nEstimate which intent from the list above fits the customer message below. "
                        'Reply with JSON only, in the form {"intent": "<intent name>"}.\n\n')


@dataclass
class Section:
    summary: str = ""
    examples: list[str] = field(default_factory=list)  # the Examples list in file order; near misses excluded


def parse_taxonomy(text: str) -> tuple[list[str], dict[str, Section]]:
    """The tie-break order, and every '## ' section's Summary line and examples."""
    order: list[str] = []
    sections: dict[str, Section] = {}
    name = ""
    in_examples = False
    for line in text.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            sections[name] = Section()
            in_examples = False
        elif not name:
            continue  # the preamble before the first section
        elif line.startswith(SUMMARY_PREFIX):
            sections[name].summary = line[len(SUMMARY_PREFIX):].strip()
        elif line.startswith("**"):
            in_examples = line.startswith("**Examples**")  # "Near misses" and every other block end the list
        else:
            order_match = ORDER_LINE.match(line)
            example_match = EXAMPLE_LINE.match(line)
            if name == TIE_BREAK_SECTION and order_match:
                order.append(order_match.group(1))
            elif in_examples and example_match:
                sections[name].examples.append(example_match.group(1))
    return order, sections


def shorten(text: str) -> str:
    """Cut an example to MAX_EXAMPLE_CHARS, marking the cut with '...'."""
    if len(text) <= MAX_EXAMPLE_CHARS:
        return text
    return text[:MAX_EXAMPLE_CHARS - 3].rstrip() + "..."


def render_block(text: str) -> str:
    """The compact block: the rules, then for each intent its one-line definition and two examples."""
    order, sections = parse_taxonomy(text)
    rules = sections.get(TIE_BREAK_SECTION)
    if rules is None or not rules.summary or not order:
        raise ValueError(f"{TAXONOMY_MD.name}: '{TIE_BREAK_SECTION}' needs a Summary line and a numbered order")
    lines = ["Customer support intents, each with a one-line definition and two real customer messages.", "",
             f"Rules: {rules.summary}", "Order: " + " > ".join(order)]
    for intent in order:
        section = sections.get(intent)
        if section is None or not section.summary or len(section.examples) < EXAMPLES_PER_INTENT:
            raise ValueError(f"{TAXONOMY_MD.name}: '{intent}' needs a Summary line and "
                             f"{EXAMPLES_PER_INTENT} examples")
        lines.append("")
        lines.append(f"{intent}: {section.summary}")
        for example in section.examples[:EXAMPLES_PER_INTENT]:
            lines.append(f'- "{shorten(example)}"')
    return "\n".join(lines) + "\n"


def intent_prompt(case: dict, block: str) -> str:
    """The block first, then the question, then the case LAST: its two latest earlier turns and the message."""
    lines = [f"Earlier {turn['role']} message: {turn['text']}" for turn in case["prior_turns"][-2:]]
    lines.append(f"Customer message: {case['customer_text']}")
    return block + ESTIMATE_INSTRUCTION + "\n".join(lines)


def intent_schema(intents: list[str]) -> dict:
    """JSON schema for exactly one intent name from the taxonomy."""
    return {"type": "object", "properties": {"intent": {"type": "string", "enum": intents}}, "required": ["intent"]}


def main() -> None:
    block = render_block(TAXONOMY_MD.read_text(encoding="utf-8"))
    PROMPT_TXT.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_TXT.write_text(block, encoding="utf-8", newline="\n")
    # num_predict=1: only the prompt matters here, and Ollama reports how many prompt tokens it evaluated.
    response = llm.complete(block, CLASSIFY_MODEL, num_predict=1)
    print(f"wrote {PROMPT_TXT}: {len(block):,} characters, {response.prompt_tokens:,} phi3 prompt tokens "
          f"(including the chat template's wrapper tokens; cache_hit={response.cache_hit})")


if __name__ == "__main__":
    main()
