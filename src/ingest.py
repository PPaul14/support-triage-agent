"""Stage 1: stream twcs.csv and rebuild one brand's support conversations as cases."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "twcs.csv"
CASES_JSONL = REPO_ROOT / "data" / "sample" / "cases.jsonl"
DEFAULT_BRAND = "SpotifyCares"
NO_PARENT = 0  # tweet ids start at 1, so 0 can mean "not a reply to anything"
TWITTER_TIME = "%a %b %d %H:%M:%S %z %Y"
DM_MENTION = re.compile(r"\bDM(s|ed|ing)?\b|(?i:direct message)")  # exact-case "DM", any-case "direct message"


@dataclass
class Turn:
    tweet_id: int
    role: str  # "customer", "brand" or "other" (a different company's account)
    author_id: str
    text: str
    created_at: str  # ISO 8601, UTC


@dataclass
class Case:
    case_id: int  # tweet_id of the customer message
    thread_id: int  # tweet_id of the thread's root
    customer_text: str
    prior_turns: list[Turn]  # root first, up to but not including the customer message
    brand_reply: str
    created_at: str
    turn_index: int  # position of the customer message in its chain, equal to len(prior_turns)


def read_links(csv_path: Path, brand: str) -> tuple[dict[int, int], list[int]]:
    """Pass 1: keep only integers - every tweet's parent id, and the ids the brand wrote."""
    parent_of: dict[int, int] = {}
    brand_ids: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tweet_id = int(row["tweet_id"])
            parent_text = row["in_response_to_tweet_id"]
            parent_of[tweet_id] = int(parent_text) if parent_text else NO_PARENT
            if row["author_id"] == brand:
                brand_ids.append(tweet_id)
    return parent_of, brand_ids


def find_root(tweet_id: int, parent_of: dict[int, int]) -> int | None:
    """Follow parent links up to the thread root. None if the chain leaves the file."""
    current = tweet_id
    # twcs.csv has no reply cycles (every chain checked, longest 649 steps), but a cycle would loop forever.
    for _ in range(len(parent_of)):  # a chain with no cycle can't take more steps than there are tweets
        if parent_of[current] == NO_PARENT:
            return current
        current = parent_of[current]
        if current not in parent_of:
            return None  # a reply to a tweet missing from the dataset: its early context is lost
    raise ValueError(f"Reply chain from tweet {tweet_id} never ends: the input has a reply cycle")


def read_threads(csv_path: Path, brand: str, parent_of: dict[int, int], roots: set[int]) -> dict[int, Turn]:
    """Pass 2: keep the full row only for tweets whose thread root is one of the brand's roots."""
    tweets: dict[int, Turn] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tweet_id = int(row["tweet_id"])
            if find_root(tweet_id, parent_of) not in roots:
                continue
            role = "customer" if row["inbound"] == "True" else "other"
            if row["author_id"] == brand:
                role = "brand"
            created_at = datetime.strptime(row["created_at"], TWITTER_TIME).isoformat()
            tweets[tweet_id] = Turn(tweet_id=tweet_id, role=role, author_id=row["author_id"],
                                    text=row["text"], created_at=created_at)
    return tweets


def build_cases(tweets: dict[int, Turn], parent_of: dict[int, int]) -> list[Case]:
    """One case per customer message with a brand reply. Ground truth: the earliest reply, lower id on a tie."""
    replies: dict[int, list[Turn]] = {}  # customer tweet id -> brand tweets that reply to it directly
    for tweet_id in sorted(tweets):
        parent_id = parent_of[tweet_id]
        if tweets[tweet_id].role == "brand" and parent_id in tweets and tweets[parent_id].role == "customer":
            replies.setdefault(parent_id, []).append(tweets[tweet_id])
    cases: list[Case] = []
    for customer_id in sorted(replies):
        earliest_reply = min(replies[customer_id], key=lambda turn: (turn.created_at, turn.tweet_id))
        chain: list[Turn] = []
        parent_id = parent_of[customer_id]
        while parent_id != NO_PARENT:  # the ancestor path only: sibling branches of a fork are never on it
            chain.append(tweets[parent_id])
            parent_id = parent_of[parent_id]
        chain.reverse()  # root first; a reply is always posted after its parent, so this is chronological
        customer = tweets[customer_id]
        cases.append(Case(case_id=customer_id, thread_id=find_root(customer_id, parent_of),
                          customer_text=customer.text, prior_turns=chain, brand_reply=earliest_reply.text,
                          created_at=customer.created_at, turn_index=len(chain)))
    cases.sort(key=lambda case: (case.created_at, case.case_id))
    return cases


def print_summary(cases: list[Case], tweets: dict[int, Turn], parent_of: dict[int, int], dropped: int) -> None:
    """Print the stage summary, with the DM share as the headline number."""
    thread_ids = set(case.thread_id for case in cases)
    tweets_per_root = Counter(find_root(tweet_id, parent_of) for tweet_id in tweets)
    brand_replies_to = Counter(parent_of[tweet_id] for tweet_id in tweets if tweets[tweet_id].role == "brand")
    multi_reply = sum(1 for case in cases if brand_replies_to[case.case_id] > 1)
    dm_replies = sum(1 for case in cases if DM_MENTION.search(case.brand_reply))
    print(f"threads:                 {len(thread_ids):,}")
    print(f"cases:                   {len(cases):,}")
    print(f"median turns per thread: {statistics.median(tweets_per_root[root] for root in thread_ids)}")
    print(f"dropped: {dropped:,} brand tweets whose chain reaches a tweet missing from the file")
    print(f"earliest reply kept for {multi_reply:,} customer messages that had more than one brand reply")
    print("=" * 72)
    print(f"  BRAND REPLIES MENTIONING DM / DIRECT MESSAGE: {dm_replies / len(cases):.1%}"
          f"  ({dm_replies:,} of {len(cases):,})")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild one brand's support threads from twcs.csv as cases.")
    parser.add_argument("--brand", default=DEFAULT_BRAND, help="author_id of the brand account")
    brand = parser.parse_args().brand
    parent_of, brand_ids = read_links(RAW_CSV, brand)
    roots = set(find_root(brand_id, parent_of) for brand_id in brand_ids)
    dropped = sum(1 for brand_id in brand_ids if find_root(brand_id, parent_of) is None)
    roots.discard(None)  # None marks a chain that leaves the file: that whole thread is dropped
    tweets = read_threads(RAW_CSV, brand, parent_of, roots)
    cases = build_cases(tweets, parent_of)
    if not cases:
        raise SystemExit(f"No cases found for brand {brand!r} in {RAW_CSV}")
    CASES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CASES_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:  # asdict also turns the nested Turn objects into plain dicts
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(f"wrote {CASES_JSONL}")
    print_summary(cases, tweets, parent_of, dropped)


if __name__ == "__main__":
    main()
