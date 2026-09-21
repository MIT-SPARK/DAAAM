# Evaluating DAAAM on OC-NaVQA

`scripts/eval_navqa.py` runs the OC-NaVQA benchmark on finished scene graphs.
Due to inherent variance in the VLM (DAM) and the LLM agent there may be small variation w.r.t. the paper results. 

## Prerequisites

- One completed pipeline run per sequence (0, 3, 4, 6, 16, 21, 22), including the post-processing steps of
  [RUNNING.md](RUNNING.md). The evaluation reads `clustered_dsg_with_summaries.json` and
  `background_objects.yaml` from each run's output directory.
- The OC-NaVQA annotations shipped in [data/](data/DATA.md).
- An OpenAI API key in the environment (`export OPENAI_API_KEY=...`) and a GPU for the retrieval encoders.

## Running

```bash
python scripts/eval_navqa.py \
  0=output/coda/out_<stamp>_seq0/clustered_dsg_with_summaries.json \
  3=output/coda/out_<stamp>_seq3/clustered_dsg_with_summaries.json \
  4=output/coda/out_<stamp>_seq4/clustered_dsg_with_summaries.json \
  6=output/coda/out_<stamp>_seq6/clustered_dsg_with_summaries.json \
  16=output/coda/out_<stamp>_seq16/clustered_dsg_with_summaries.json \
  21=output/coda/out_<stamp>_seq21/clustered_dsg_with_summaries.json \
  22=output/coda/out_<stamp>_seq22/clustered_dsg_with_summaries.json
```

## Results

`output/oc_navqa/navqa_eval_seq_<sequence>_<seed>_<timestamp>.json` holds every answer with its reasoning,
tool-call trace, token usage and error, plus the run's metadata. `output/oc_navqa/aggregated_<timestamp>.json`
and the printed table give the metrics pooled over all (question, seed) samples.

## Reference numbers

Two annotation files ship in [data/](data/DATA.md): `data/oc-navqa_data-legacy.csv` is the version used for the
experiments in the paper; `data/oc-navqa_data.csv` is the current version, in which annotation errors were fixed
and ambiguous questions were disambiguated after the paper's experiments. The script evaluates on the current
version (`CORRECTED_CSV` in the script); point it at the legacy file to compare with the paper.

Three-seed results on all seven sequences:

| | Binary accuracy | Spatial error [m] | Temporal error [min] |
|---|---|---|---|
| DAAAM (legacy annotations), as reported in the paper | 0.711 | 41.75 | 1.79 |
| DAAAM (fixed annotations), this release, `gpt-5-mini` | 0.726 | 36.32 | 1.45 |
| DAAAM (fixed annotations), this release, `gpt-5.6-luna` | 0.751 | 34.55 | 1.32 |

The paper's numbers were obtained with the private implementation on the legacy annotations; this release
evaluated on the same legacy annotations reproduces them within seed variance (three seeds: 0.721, 40.14 m,
2.10 min temporal). The grounding VLM, the region summaries and the agent are non-deterministic, so single runs
scatter around these values; compare averages over the three seeds.
