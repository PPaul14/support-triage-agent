# Data license

The code in this repository is MIT licensed (see [LICENSE](../LICENSE)). The
data is not.

Source: the Customer Support on Twitter dataset,
[thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
on Kaggle, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- Quoted dataset excerpts in this repository - the example customer messages
  in `docs/taxonomy.md` and `artifacts/classifier_prompt.txt` - remain under
  CC BY-NC-SA 4.0: attribute them to
  thoughtvector/customer-support-on-twitter, use them only non-commercially,
  and share adaptations under the same licence.
- The raw file and everything derived from it with message text
  (`data/raw/`, `data/sample/cases*.jsonl`, `artifacts/clusters.md`) are not
  committed. Download `twcs.csv` from the Kaggle page into `data/raw/` and
  rebuild them with the pipeline.
- The test fixtures in `tests/fixtures/` are hand-written and are not taken
  from the dataset.
