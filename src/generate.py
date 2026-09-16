"""Stage 6 (draft): llama3 writes the reply from the case and its retrieved precedents.

The rules go in the system message; the case follows in delimited sections, the precedents labelled as other
customers' conversations. A draft is rejected, and retried once with the reason, when it copies ECHO_WORDS words
in a row from the prompt (a precedent's own reply excepted: reusing the brand's wording is the point) or names a
refund, amount or timeline that no precedent states."""

import hashlib
import re
from dataclasses import dataclass

from src import llm
from src.retrieve import Hit

DRAFT_MODEL = "llama3:8b-instruct-q4_0"
NUM_PREDICT = 80
MAX_CHARS = 280
ECHO_WORDS = 8
MIN_SHARED_WORDS = 3  # below this many shared words of 4+ letters, a draft is said to use no precedent
SYSTEM = ("You write replies for Spotify's customer support team on Twitter. Rules:\n"
          "1. Output ONLY the reply text: no heading, label, quotation marks, note or explanation.\n"
          "2. Under 280 characters, friendly and brief. No signature and no @handles.\n"
          "3. The OTHER CUSTOMERS' PAST CASES are other people's conversations, never this customer's: do not "
          "refer to them as anything this customer said or was told. Take facts and steps only from the brand's "
          "replies there. Never mention a refund, an amount of money, a date or a timeline unless one of those "
          "replies states it.\n"
          "4. If there are no past cases, or none fits, acknowledge the problem and ask for the details needed "
          "to help.")
RETRY_NOTE = "\n\nYour previous reply was rejected because it {problems}. Write a new reply. Output ONLY the reply text."
SPECIFIC = re.compile(r"[$£€]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:%|percent|dollars?|pounds?|euros?)\b|\brefund"
                      r"|\b\d+\s?(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b"
                      r"|\btomorrow\b|\bnext (?:week|month)\b", re.IGNORECASE)


@dataclass
class Attempt:
    prompt_sha256: str  # of the system text and the prompt together
    response: llm.LLMResponse
    text: str  # the reply after tidy()
    trimmed: bool
    rejected_for: list[str]  # empty when the draft passed its checks


@dataclass
class Draft:
    text: str  # the last attempt's text, even when it was rejected
    accepted: bool
    precedent_id: int | None  # the precedent the reply draws on (attribute), or None
    shared_words: int
    attempts: list[Attempt]


def build_prompt(case: dict, intent: str, hits: list[Hit]) -> str:
    """The per-case sections, each under a delimited heading, then the reply instruction repeated."""
    lines = ["=== PREDICTED INTENT ===", intent, "", "=== OTHER CUSTOMERS' PAST CASES (not this conversation) ==="]
    if not hits:
        lines.append("(none)")
    for number, hit in enumerate(hits, start=1):
        lines.append(f"[{number}] Another customer wrote: {hit.precedent.customer_text}")
        lines.append(f"[{number}] The brand replied: {hit.precedent.brand_reply}")
    lines += ["", "=== THIS CONVERSATION: EARLIER TURNS ==="]
    if not case["prior_turns"]:
        lines.append("(none)")
    for turn in case["prior_turns"][-2:]:
        lines.append(f"{turn['role']}: {turn['text']}")
    lines += ["", "=== THIS CONVERSATION: THE CUSTOMER MESSAGE TO ANSWER ===", case["customer_text"], "",
              "Write the reply to this customer's message now. Output ONLY the reply text."]
    return "\n".join(lines)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(text: str) -> set[tuple[str, ...]]:
    tokens = words(text)
    return set(tuple(tokens[start:start + ECHO_WORDS]) for start in range(len(tokens) - ECHO_WORDS + 1))


def copies_prompt(text: str, prompt: str, hits: list[Hit]) -> bool:
    """A section marker, or an ECHO_WORDS-word run shared with the rules or the prompt that no precedent reply has."""
    if "===" in text:
        return True
    allowed: set[tuple[str, ...]] = set()
    for hit in hits:
        allowed |= ngrams(hit.precedent.brand_reply)
    copied = ngrams(text) & ngrams(SYSTEM + "\n" + prompt)
    return len(copied - allowed) > 0


def problems(text: str, prompt: str, hits: list[Hit]) -> list[str]:
    """Why a draft cannot be used; empty when it can."""
    if not text:
        return ["was empty"]
    found = []
    if copies_prompt(text, prompt, hits):
        found.append("copied text from the prompt")
    evidence = " ".join(hit.precedent.brand_reply for hit in hits).lower()
    for match in SPECIFIC.finditer(text):
        if match.group(0).lower() not in evidence:
            found.append(f"mentions '{match.group(0)}', which no past reply states")
    return found


def tidy(text: str, ran_out: bool) -> tuple[str, bool]:
    """Strip wrapping quotes; if too long or cut off, end at the last full sentence (else word). Returns (text, cut)."""
    text = text.strip().strip('"').strip()
    if len(text) <= MAX_CHARS and not ran_out:
        return text, False
    window = text[:MAX_CHARS]
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", window)]
    if sentence_ends:
        cut = window[:sentence_ends[-1]]
    elif len(text) > MAX_CHARS:
        cut = window[:window.rfind(" ")].rstrip()
    else:
        cut = window
    return cut, cut != text


def attribute(text: str, hits: list[Hit]) -> tuple[int | None, int]:
    """The precedent whose reply shares the most words of 4+ letters with the draft, and how many it shares.
    Ties go to the more similar precedent; under MIN_SHARED_WORDS the draft is said to use none."""
    draft_words = set(word for word in words(text) if len(word) >= 4)
    best_id, best_shared = None, 0
    for hit in hits:
        shared = len(draft_words & set(words(hit.precedent.brand_reply)))
        if shared > best_shared:
            best_id, best_shared = hit.precedent.case_id, shared
    if best_shared < MIN_SHARED_WORDS:
        return None, best_shared
    return best_id, best_shared


def draft(case: dict, intent: str, hits: list[Hit]) -> Draft:
    """Draft and check; if the draft fails a check, retry once with the reasons appended to the prompt."""
    base = build_prompt(case, intent, hits)
    prompt = base
    attempts: list[Attempt] = []
    for _ in range(2):
        response = llm.complete(prompt, DRAFT_MODEL, system=SYSTEM, num_predict=NUM_PREDICT)
        text, trimmed = tidy(response.text, response.completion_tokens >= NUM_PREDICT)
        found = problems(text, base, hits)
        prompt_sha256 = hashlib.sha256((SYSTEM + "\n" + prompt).encode("utf-8")).hexdigest()
        attempts.append(Attempt(prompt_sha256, response, text, trimmed, found))
        if not found:
            break
        prompt = base + RETRY_NOTE.format(problems="; ".join(found))
    final = attempts[-1]
    precedent_id, shared = attribute(final.text, hits)
    return Draft(final.text, not final.rejected_for, precedent_id, shared, attempts)
