# Evidence-Support Classification with Local LLMs

**NLP course project — LAQuer-inspired extension using SPARK evidence**

This repository contains a reproducible evaluation of six locally hosted large language models (LLMs) on **atomic-fact evidence-support classification**. For each atomic fact, a model receives five retrieved source-evidence windows and predicts whether the combined evidence supports the complete claim.

The project is motivated by *LAQuer (LLMs as Attribution Queryers)*, but **does not reproduce LAQuer's original attribution-localization scores or execute its end-to-end span-localization pipeline**. Instead, it investigates a complementary task: assessing the degree of support provided by retrieved evidence. Integrating such a verifier after attribution retrieval is a possible future extension, not an implemented feature of this submission.

## Quick start: reproduce the reported evaluation

**No Ollama installation or model download is needed for this step.** The six original prediction CSVs are included so the evaluation can be reproduced offline.

From the submission root, run:

```powershell
python evaluation/consolidate_attribution_evaluation.py
```

The script checks that all six model files contain the same 57 labeled facts and the expected frozen prompt version, then regenerates the metrics, confusion matrices, comparison table, Markdown report, and evaluation manifest under `evaluation/model_evaluation/`.

**Expected summary (rounded):**

| Model | Correct / 57 | Accuracy | Three-class macro-F1 |
|---|---:|---:|---:|
| Qwen3 1.7B | 35 | 61.4% | 32.1% |
| Qwen3 8B | 42 | 73.7% | 72.0% |
| Llama 3.2 3B | 16 | 28.1% | 28.3% |
| Llama 3.1 8B | 40 | 70.2% | 57.0% |
| Gemma 3 4B | 36 | 63.2% | 30.1% |
| Gemma 3 12B | 44 | 77.2% | 66.2% |

Gemma 3 12B achieves the highest accuracy; Qwen3 8B achieves the highest three-class macro-F1. These results describe the supplied evaluation set and should not be interpreted as population-wide performance estimates.

## Repository layout

```text
codebase extension/
├── README.md
├── requirements.txt                  # Optional; no third-party Python packages required by core scripts
├── .gitignore
├── data/
│   ├── attribution_57_facts_reference.csv
│   └── attribution_300_candidates_provisional_consolidated.csv
├── scripts/
│   ├── run_attribution_ollama_pilot_v2.py
│   └── run_attribution_full_v2.py
├── evaluation/
│   ├── consolidate_attribution_evaluation.py
│   └── model_evaluation/
│       ├── attribution_qwen3_1_7b_57facts_v2.csv
│       ├── attribution_qwen3_8b_57facts_v2.csv
│       ├── attribution_llama3_2_3b_57facts_v2.csv
│       ├── attribution_llama3_1_8b_57facts_v2.csv
│       ├── attribution_gemma3_4b_57facts_v2.csv
│       └── attribution_gemma3_12b_57facts_v2.csv
└── model_evaluation/                # Created by fresh full-model inference
```

The six CSVs under `evaluation/model_evaluation/` are the **saved experiment results**. The root-level `model_evaluation/` is a **separate destination for newly generated inference results**; do not substitute fresh results for the saved files when reproducing the reported experiment.

## Requirements

- Python 3.12 (the development environment); core scripts use only the Python standard library.
- For **fresh inference only**: a running [Ollama](https://ollama.com/) server and at least one locally installed model.
- Enough RAM/VRAM and disk space for the selected Ollama model; requirements vary by model and quantization.

Check Python and Ollama:

```powershell
python --version
ollama list
```

For example, install the smallest evaluated model if it is not already available:

```powershell
ollama pull qwen3:1.7b
```

Ollama's API must be reachable at `http://127.0.0.1:11434` (the scripts' default). If it is not running, start Ollama using the method appropriate for your installation.

## Dataset and task

The experiments use SPARK-derived multi-document source material and a fixed set of atomic facts and retrieved evidence windows.

- **Reference evaluation:** 57 manually adjudicated atomic facts, selected from 60 inspected facts; three excluded facts are not scored.
- **Reference label counts:** 36 `SUPPORTED`, 14 `PARTIALLY_SUPPORTED`, 7 `INSUFFICIENT`, and zero `CONTRADICTED`.
- **Candidate input file:** 300 rows representing five ranked evidence windows for each of the original 60 sampled facts. The inference code selects the five windows corresponding to each of the 57 scored facts.
- **Model inputs:** one atomic fact and five ranked retrieved evidence windows, considered **together**.
- **Predictions:** one label and a short explanation per fact.

The four possible output labels mean:

| Label | Meaning |
|---|---|
| `SUPPORTED` | The combined evidence establishes every substantive component of the claim. |
| `PARTIALLY_SUPPORTED` | At least one substantive component is established, but another remains unverified. |
| `CONTRADICTED` | The evidence explicitly conflicts with a central component of the claim. |
| `INSUFFICIENT` | The evidence establishes no substantive component of the claim. |

The exact frozen system instructions and user-prompt construction are defined in `scripts/run_attribution_ollama_pilot_v2.py` and imported by the full-run script. The experiment uses prompt version `attribution_top5_v2`, temperature `0.0`, structured JSON output, and a generation cap of 420 tokens.

## Run a small live-inference pilot

From the submission root:

```powershell
python scripts/run_attribution_ollama_pilot_v2.py --model qwen3:1.7b --count 3
```

This tests three facts and prints their predicted labels, reference labels, timing, and any quality warnings. The pilot writes a CSV output; by default, its output path is relative to the pilot script's directory.

**Path configuration:** This command assumes the packaged pilot script has been updated to read its two input CSVs from the root-level `data/` folder. If using the original, unmodified pilot script, specify the input paths explicitly:

```powershell
python scripts/run_attribution_ollama_pilot_v2.py `
  --model qwen3:1.7b `
  --count 3 `
  --facts data/attribution_57_facts_reference.csv `
  --candidates data/attribution_300_candidates_provisional_consolidated.csv
```

## Run a full model (57 facts)

From the submission root:

```powershell
python scripts/run_attribution_full_v2.py --model qwen3:1.7b
```

This command assumes the packaged full-run script has been updated to read from `data/` at the repository root and write to root-level `model_evaluation/`, as described above. The full-run script:

1. Validates the 57 scored facts and the five ranked evidence windows for each fact.
2. Calls Ollama's `/api/chat` endpoint with the frozen prompt and JSON response schema.
3. Writes each result to a model-specific CSV immediately.
4. Resumes previously saved successful predictions if the same command is run again.
5. Prints the valid prediction count, exact-match count, label distribution, and quality flags.

To run all six models sequentially, after downloading all of them:

```powershell
python scripts/run_attribution_full_v2.py --all-models
```

Supported model identifiers are `qwen3:1.7b`, `qwen3:8b`, `llama3.2:3b`, `llama3.1:8b`, `gemma3:4b`, and `gemma3:12b`.

**Important:** The full-run script uses `--directory` as its project-root setting in the packaged layout. Its default should be the parent of `scripts/`. New inference outputs are not automatically included in the saved-results evaluation. The evaluation script reads only from `evaluation/model_evaluation/`.

## Evaluation procedure

Run:

```powershell
python evaluation/consolidate_attribution_evaluation.py
```

For each model, the evaluation calculates:

- **Accuracy:** fraction of the 57 facts assigned their reference label.
- **Per-class precision, recall, and F1:** for `SUPPORTED`, `PARTIALLY_SUPPORTED`, and `INSUFFICIENT`.
- **Three-class macro-F1:** arithmetic mean of F1 for those three reference classes.
- **Confusion matrix:** three reference-label rows by four possible predicted-label columns.
- **Development/untouched breakdown:** 14 previously inspected development cases versus 43 previously untouched cases.

Predictions of `CONTRADICTED` count as errors when the reference label is different. Because the adjudicated set has no `CONTRADICTED` reference cases, that class is not included in the reported three-class macro-F1.

The script also writes a 342-row consolidated prediction CSV, a per-fact model comparison, misclassifications, a Markdown report, and a JSON manifest. The input CSVs include raw model responses and timing/quality information.

## Reproducibility and limitations

- **Offline reproduction vs. rerunning inference:** Offline evaluation reproduces the reported metrics from frozen prediction files. Fresh Ollama inference may differ with model builds, quantization, hardware, or runtime versions, even at temperature zero.
- **Small evaluation sample:** Results are based on 57 adjudicated facts, not a representative benchmark of all SPARK claims.
- **Development overlap:** Fourteen facts were inspected during prompt development; the evaluation reports the 43 untouched facts separately to make this distinction visible.
- **Missing reference class:** There are no adjudicated `CONTRADICTED` examples, so recall for that class cannot be assessed.
- **Retrieval dependency:** Models classify the five supplied windows; a missing or poorly ranked supporting passage can affect the apparent support level.
- **Local timing:** Inference times reflect the machine and model configuration used, not hardware-independent speed comparisons.
- **Scope:** This repository evaluates support classification, not LAQuer's original span-retrieval or attribution-localization metric.

## Troubleshooting

**`FileNotFoundError` for the reference or candidate CSV:** Confirm both input files are under `data/`. The original pilot script looks beside itself unless its defaults were updated or `--facts`/`--candidates` were passed explicitly. The full-run script likewise needs its `load_inputs()` path updated for the packaged layout.

**`ModuleNotFoundError: run_attribution_ollama_pilot_v2`:** Keep the pilot and full-run scripts together in `scripts/`.

**Connection refused at port 11434:** Start Ollama and confirm its local API is available.

**Model not found:** Run `ollama list`, then `ollama pull <model-name>` for the requested model.

**Unexpected result differences:** Confirm the model identifier, Ollama version, frozen prompt, input CSVs, and generation settings. Use the saved-results evaluation for exact reproduction of the published metrics.

## Provenance

This submission builds on the LAQuer research direction and SPARK multi-document summarization resources. Consult the accompanying course project report for full academic references, dataset provenance, experimental design, and interpretation. The model-specific CSVs and evaluation manifest provide the immediate computational record for the results summarized here.
