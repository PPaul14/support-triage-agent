"""Stage 3 (exploration): cluster a sample of customer messages so the intent taxonomy comes from the data.

The report is deliberately unnamed: intents get names only after a human has read the clusters.
"""

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = REPO_ROOT / "data" / "sample" / "cases_clean.jsonl"
EMBEDDINGS_DIR = REPO_ROOT / "artifacts" / "embeddings"
REPORT_MD = REPO_ROOT / "artifacts" / "clusters.md"  # quotes tweets: gitignored
SUMMARY_MD = REPO_ROOT / "artifacts" / "clusters_summary.md"  # sizes and terms only: committed
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned Hugging Face commit: same weights for all
SAMPLE_SIZE = 4000
N_CLUSTERS = 25
SEED = 0
NEAREST = 8
TOP_TERMS = 10


@dataclass
class Cluster:
    number: int  # 1 is the largest cluster; KMeans' own label numbers carry no meaning
    size: int
    terms: list[str]  # most distinctive first
    nearest: list[str]  # the messages closest to the centroid, closest first


def sample_messages(path: Path) -> list[tuple[int, str]]:
    """(case_id, text) pairs, one per near-duplicate group, then a seeded sample. @USER is stripped from text."""
    pool: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            # A group is represented by its smallest case_id, which is also its dup_group_id.
            if case["dup_group_id"] is None or case["dup_group_id"] == case["case_id"]:
                text = " ".join(token for token in case["customer_text"].split() if token != "@USER")
                pool.append((case["case_id"], text))
    pool.sort()  # sample from a fixed order, so the file's row order cannot change the sample
    return sorted(random.Random(SEED).sample(pool, SAMPLE_SIZE))


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, one row per text. Only texts missing from the disk cache are computed."""
    # Model and revision are in the file name, so the key only needs the text.
    cache_path = EMBEDDINGS_DIR / f"all-MiniLM-L6-v2-{MODEL_REVISION[:12]}.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as stored:  # closed at once: Windows cannot replace a file that is still open
            cached = dict(zip(stored["keys"].tolist(), stored["vectors"]))
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    missing = {key: text for key, text in zip(keys, texts) if key not in cached}
    if missing:
        model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        missing_keys = sorted(missing)
        vectors = model.encode([missing[key] for key in missing_keys], batch_size=64, normalize_embeddings=True)
        cached.update(zip(missing_keys, vectors))
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        all_keys = sorted(cached)
        temp_path = cache_path.with_suffix(".tmp.npz")  # temp file + rename: a killed run never leaves half a cache
        np.savez(temp_path, keys=np.array(all_keys), vectors=np.stack([cached[key] for key in all_keys]))
        temp_path.replace(cache_path)
    print(f"embeddings: {len(texts) - len(missing):,} from cache, {len(missing):,} computed")
    return np.stack([cached[key] for key in keys])


def describe_clusters(texts: list[str], vectors: np.ndarray) -> list[Cluster]:
    """KMeans, then for each cluster: its size, the messages nearest its centroid and its distinctive terms."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(vectors)
    distances = kmeans.transform(vectors)  # distance from every message to every centroid
    members: list[list[int]] = [[] for _ in range(N_CLUSTERS)]
    for index, label in enumerate(labels):
        members[label].append(index)
    # Class-based TF-IDF: each cluster's messages joined into one document, scored against the other clusters.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    scores = vectorizer.fit_transform([" ".join(texts[i] for i in group) for group in members]).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    order = sorted(range(N_CLUSTERS), key=lambda label: (-len(members[label]), label))  # largest first
    clusters: list[Cluster] = []
    for number, label in enumerate(order, start=1):
        term_order = sorted(range(len(vocabulary)), key=lambda i: (-scores[label][i], vocabulary[i]))
        nearest = sorted(members[label], key=lambda i: (distances[i][label], i))
        clusters.append(Cluster(number=number, size=len(members[label]),
                                terms=[str(vocabulary[i]) for i in term_order[:TOP_TERMS]],
                                nearest=[texts[i] for i in nearest[:NEAREST]]))
    return clusters


def render_report(clusters: list[Cluster]) -> str:
    """The raw cluster report as markdown. Clusters are numbered, never named."""
    lines = ["# Raw clusters (unnamed)", "",
             f"{SAMPLE_SIZE:,} customer messages from `cases_clean.jsonl`, one per near-duplicate group, "
             f"sampled with seed {SEED}; `@USER` stripped. Embedded with `{MODEL}` at revision "
             f"`{MODEL_REVISION[:12]}`, clustered with KMeans (k={N_CLUSTERS}, seed {SEED}). "
             "Clusters are numbered by size, largest first. Terms are class-based TF-IDF against the other "
             "clusters; messages are the ones nearest the centroid.", ""]
    for cluster in clusters:
        lines.append(f"## Cluster {cluster.number}: {cluster.size} messages ({cluster.size / SAMPLE_SIZE:.1%})")
        lines.append("")
        lines.append("Distinctive terms: " + ", ".join(cluster.terms))
        lines.append("")
        for rank, text in enumerate(cluster.nearest, start=1):
            lines.append(f"{rank}. {text}")
        lines.append("")
    return "\n".join(lines)


def render_summary(clusters: list[Cluster]) -> str:
    """The redacted report that goes into git: sizes and distinctive terms, no message text."""
    lines = ["# Cluster summary (redacted)", "",
             f"Same run as `artifacts/clusters.md` ({SAMPLE_SIZE:,} messages, KMeans k={N_CLUSTERS}, seed {SEED}), "
             "without the example messages: that file quotes tweets from a CC BY-NC-SA dataset and is not "
             "committed. `python -m src.taxonomy` rebuilds both.", "",
             "| cluster | messages | share | distinctive terms |", "|---:|---:|---:|---|"]
    for cluster in clusters:
        lines.append(f"| {cluster.number} | {cluster.size} | {cluster.size / SAMPLE_SIZE:.1%} | {', '.join(cluster.terms)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # a Windows pipe defaults to cp1252, which cannot print emoji
    sample = sample_messages(CLEAN_JSONL)
    texts = [text for _, text in sample]
    clusters = describe_clusters(texts, embed(texts))
    report = render_report(clusters)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    SUMMARY_MD.write_text(render_summary(clusters), encoding="utf-8")
    print(report)
    print(f"wrote {REPORT_MD} and {SUMMARY_MD}")


if __name__ == "__main__":
    main()
