# Credits

Author: Priyanshi Paul <ppriyanshi1177@gmail.com>. The code in this repository
is MIT licensed; the dataset excerpts it quotes are not (see Data, below).

## Data

**Customer Support on Twitter**, published by Stuart Axelbrooke (thoughtvector)
on Kaggle:
[kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

- **Licence: CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike.
  Full terms: [creativecommons.org/licenses/by-nc-sa/4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The corpus itself is **not redistributed here**. `data/raw/twcs.csv` must be
  downloaded from Kaggle; `.gitignore` keeps it, the derived case files, the
  traces and the LLM cache out of the repository.
- The only message text committed is **46 quoted customer messages** used as
  the classifier's few-shot examples, in `docs/taxonomy.md` and the block
  rendered from it, `artifacts/classifier_prompt.txt`. Those excerpts remain
  under CC BY-NC-SA 4.0. See [../data/DATA_LICENSE.md](../data/DATA_LICENSE.md).
- This project is a non-commercial evaluation exercise, consistent with the
  licence's NC term.

## Models

All inference is local, through [Ollama](https://ollama.com) 0.34.2. No hosted
model API is called anywhere in this project.

| Role | Model | Publisher | Licence |
|---|---|---|---|
| classify, escalation layer 4 | `phi3:3.8b-mini-128k-instruct-q4_0` | Microsoft | MIT |
| draft replies | `llama3:8b-instruct-q4_0` | Meta | Llama 3 Community License |
| judge A (cross-judge sample) | `mistral:7b-instruct-v0.3-q4_K_M` | Mistral AI | Apache 2.0 |
| judge B (every reply) | `qwen2.5:7b-instruct-q4_K_M` | Alibaba Cloud | Apache 2.0 |
| embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers / UKP Lab | Apache 2.0 |

Manifest digests for the four Ollama models are in
[../README.md](../README.md#models); MiniLM is pinned to Hugging Face commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Libraries

Python 3.12.10. Every direct dependency is pinned in
[../requirements.txt](../requirements.txt), and each one does real work here:

| Library | Version | What it does in this project |
|---|---|---|
| requests | 2.34.2 | the HTTP calls to Ollama, in `src/llm.py` |
| jsonschema | 4.26.0 | validates every structured model reply before it is cached |
| numpy | 2.5.3 | embedding matrices, cosine search, bootstrap resampling |
| scikit-learn | 1.9.1 | KMeans for the taxonomy, TF-IDF and logistic regression for B1 and the precedent intents |
| sentence-transformers | 6.0.1 | MiniLM embeddings for clustering and retrieval |
| datasketch | 2.0.0 | MinHash LSH for near-duplicate detection |
| ftfy | 6.3.1 | repairs mojibake in the raw tweet text |
| langdetect | 1.0.9 | the language tag on each case |
| pytest | 9.1.1 | the test suite, including the leak-prevention tests |

## Prior art and references

- **ROUGE-L**: Lin, C.-Y. (2004), *ROUGE: A Package for Automatic Evaluation of
  Summaries*. Implemented directly in `eval/reference.py`; used as a diagnostic
  only, never as a headline.
- **Wilson score interval**: Wilson, E. B. (1927), used for the audit agreement
  in `eval/label_stats.py`.
- **Rule of three**: the 95% upper bound of about 3/n when zero events are
  observed, used for the SEVERE harm bound in `REPORT.md` Section 5.
- **MinHash / LSH**: Broder, A. (1997), via datasketch.
