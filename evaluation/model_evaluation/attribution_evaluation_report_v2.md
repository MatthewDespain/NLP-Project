# SPARK attribution evaluation — six Ollama models

## Evaluation setup

All models used the same frozen `attribution_top5_v2` prompt and five evidence windows per fact. Reference: 57 adjudicated facts (36 SUPPORTED, 14 PARTIALLY_SUPPORTED, 7 INSUFFICIENT); 3 excluded facts were not scored. Six models × 57 facts = 342 predictions. The 14 previously inspected development examples are separated from 43 previously untouched examples.

## Model results

| Model | Correct / 57 | Accuracy | Macro-F1 (3 classes) | Untouched correct / 43 | Mean seconds / fact | Warnings |
|---|---:|---:|---:|---:|---:|---:|
| gemma3:12b | 44/57 | 77.2% | 66.2% | 36/43 (83.7%) | 58.40 | 1 |
| qwen3:8b | 42/57 | 73.7% | 72.0% | 32/43 (74.4%) | 36.95 | 3 |
| llama3.1:8b | 40/57 | 70.2% | 57.0% | 30/43 (69.8%) | 33.21 | 2 |
| gemma3:4b | 36/57 | 63.2% | 30.1% | 29/43 (67.4%) | 17.65 | 1 |
| qwen3:1.7b | 35/57 | 61.4% | 32.1% | 29/43 (67.4%) | 10.83 | 4 |
| llama3.2:3b | 16/57 | 28.1% | 28.3% | 11/43 (25.6%) | 14.78 | 1 |

## Per-class metrics

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| qwen3:1.7b | SUPPORTED | 70.2% | 91.7% | 79.5% |
| qwen3:1.7b | PARTIALLY_SUPPORTED | 20.0% | 14.3% | 16.7% |
| qwen3:1.7b | INSUFFICIENT | 0.0% | 0.0% | 0.0% |
| qwen3:8b | SUPPORTED | 90.0% | 75.0% | 81.8% |
| qwen3:8b | PARTIALLY_SUPPORTED | 47.6% | 71.4% | 57.1% |
| qwen3:8b | INSUFFICIENT | 83.3% | 71.4% | 76.9% |
| llama3.2:3b | SUPPORTED | 0.0% | 0.0% | 0.0% |
| llama3.2:3b | PARTIALLY_SUPPORTED | 25.5% | 100.0% | 40.6% |
| llama3.2:3b | INSUFFICIENT | 100.0% | 28.6% | 44.4% |
| llama3.1:8b | SUPPORTED | 76.2% | 88.9% | 82.1% |
| llama3.1:8b | PARTIALLY_SUPPORTED | 46.2% | 42.9% | 44.4% |
| llama3.1:8b | INSUFFICIENT | 100.0% | 28.6% | 44.4% |
| gemma3:4b | SUPPORTED | 64.8% | 97.2% | 77.8% |
| gemma3:4b | PARTIALLY_SUPPORTED | 50.0% | 7.1% | 12.5% |
| gemma3:4b | INSUFFICIENT | 0.0% | 0.0% | 0.0% |
| gemma3:12b | SUPPORTED | 78.3% | 100.0% | 87.8% |
| gemma3:12b | PARTIALLY_SUPPORTED | 57.1% | 28.6% | 38.1% |
| gemma3:12b | INSUFFICIENT | 100.0% | 57.1% | 72.7% |

## Confusion matrices

Each row is a reference class; columns are predicted SUPPORTED / PARTIALLY_SUPPORTED / INSUFFICIENT / CONTRADICTED.

### qwen3:1.7b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 33 | 3 | 0 | 0 |
| PARTIALLY_SUPPORTED | 12 | 2 | 0 | 0 |
| INSUFFICIENT | 2 | 5 | 0 | 0 |

### qwen3:8b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 27 | 9 | 0 | 0 |
| PARTIALLY_SUPPORTED | 3 | 10 | 1 | 0 |
| INSUFFICIENT | 0 | 2 | 5 | 0 |

### llama3.2:3b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 0 | 36 | 0 | 0 |
| PARTIALLY_SUPPORTED | 0 | 14 | 0 | 0 |
| INSUFFICIENT | 0 | 5 | 2 | 0 |

### llama3.1:8b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 32 | 4 | 0 | 0 |
| PARTIALLY_SUPPORTED | 8 | 6 | 0 | 0 |
| INSUFFICIENT | 2 | 3 | 2 | 0 |

### gemma3:4b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 35 | 1 | 0 | 0 |
| PARTIALLY_SUPPORTED | 13 | 1 | 0 | 0 |
| INSUFFICIENT | 6 | 0 | 0 | 1 |

### gemma3:12b

| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |
|---|---:|---:|---:|---:|
| SUPPORTED | 36 | 0 | 0 | 0 |
| PARTIALLY_SUPPORTED | 10 | 4 | 0 | 0 |
| INSUFFICIENT | 0 | 3 | 4 | 0 |

## Methodological notes

- The reference dataset has no CONTRADICTED examples. Predictions of CONTRADICTED are counted as errors, and three-class macro-F1 averages F1 over the three reference classes. Undefined precision for unpredicted classes is set to zero.
- These are descriptive results from a small, stratified sample; the 14 development cases were inspected during prompt development. Untouched-subset scores are reported separately, but the sample is not a population-representative benchmark.
- A quality flag marks a possibly truncated explanation, not necessarily an invalid label. Raw model responses are retained in the 342-row predictions CSV.
- The per-fact comparison table enables disagreement analysis; classification correctness alone does not independently validate the human reference or source evidence.
- Inference times reflect the local execution environment and should not be treated as hardware-independent benchmarks.
