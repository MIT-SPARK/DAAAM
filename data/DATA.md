## OC-NaVQA annotations

In ./oc-navqa_data.csv, we provide OC-NaVQA the corrected annotations for the NaVQA dataset (see [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr)).

Two versions of the annotation file are provided:

- `./oc-navqa_data-legacy.csv`: the annotations used for the experiments in the paper.
- `./oc-navqa_data.csv`: the current annotations. After the paper's experiments, a number of annotation errors highlighted by the community were fixed and ambiguous questions were disambiguated. Use this version for new experiments; see [EVAL.md](../EVAL.md) for reference numbers on both versions.

The annotations incorporate three major differences:

- They expand the horizon of the questions: the context window is always from the beginning of the sequence until the `current time` .
- They correct position annotations using the ground truth annotations of the UT Campus Object Dataset [CODa](https://amrl.cs.utexas.edu/coda/) such that position annotations are where the object _is_ rather than from which pose it was observed.
- They resolve ambiguities in the framing of the questions.

Please refer to [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr) to see how the data can be used. It can simply be replaced with the [data.csv](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/remembr/data/navqa/data.csv) of the remembr NaVQA dataset.

To process the data into the remembr `question_jsons`, refer to the script in our fork of remembr: https://github.com/nicogorlo/remembr/blob/main/remembr/scripts/question_scripts/form_question_jsons_fullseq.py and the preprocessing script https://github.com/nicogorlo/remembr/blob/main/remembr/scripts/preprocess_coda.py . 

IMPORTANT: Note that while the NaVQA used the `dense` poses from the CODa dataset, we use the `dense_global` poses, as the ground-truth bounding box annotations in the CODa dataset are defined in the global coordinates.

## OC-NaVQA question files

`./oc-navqa/questions/{sequence}/human_qa_fullseq_v2_seconds.json` contains the questions in the form consumed by
`scripts/eval_navqa.py` (see [EVAL.md](../EVAL.md)), one file per CODa sequence (0, 3, 4, 6, 16, 21, 22; 30 questions each). At evaluation time the question text and the position ground truth are taken from `./oc-navqa_data.csv`, which overrides the JSON. The JSON therefore only supplies the time-related fields and the binary, text, time and duration answers.

## Attribution

When using this data, for courtesy please also cite [ReMEmbR](https://arxiv.org/abs/2409.13682), as OC-NaVQA is a derivative of their data with some changes.