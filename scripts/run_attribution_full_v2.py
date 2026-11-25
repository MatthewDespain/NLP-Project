#!/usr/bin/env python3
"""Run frozen attribution_top5_v2 on all 57 scored facts, one Ollama model at a time.

Place beside run_attribution_ollama_pilot_v2.py and the two input CSVs.
Uses only the Python standard library. Existing successful predictions are resumed.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_attribution_ollama_pilot_v2 as v2

MODELS = ('qwen3:1.7b', 'qwen3:8b', 'llama3.2:3b', 'llama3.1:8b', 'gemma3:4b', 'gemma3:12b')
FIELDS = ('sample_id', 'fact_id', 'similarity_stratum', 'model', 'prompt_version',
          'reference_label', 'predicted_label', 'match', 'explanation', 'raw_response',
          'error', 'quality_flags', 'done_reason', 'elapsed_seconds')


def load_inputs(folder):
    facts = v2.read_csv(folder / 'data' / v2.FACTS)
    candidates = v2.read_csv(folder / 'data' / v2.CANDIDATES)
    if len(facts) != 57 or len({r['fact_id'] for r in facts}) != 57:
        raise ValueError('Expected 57 unique scored reference facts')
    if len(candidates) != 300:
        raise ValueError('Expected 300 original candidate windows')
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row['fact_id']].append(row)
    for fact in facts:
        group = sorted(grouped[fact['fact_id']], key=lambda x: int(x['rank']))
        if [int(x['rank']) for x in group] != [1, 2, 3, 4, 5]:
            raise ValueError(f'Expected five ranked windows for {fact["sample_id"]}')
        if any(x['fact'] != fact['fact'] for x in group):
            raise ValueError(f'Fact text mismatch for {fact["sample_id"]}')
        grouped[fact['fact_id']] = group
    return facts, grouped


def save(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def run_model(model, args, facts, grouped):
    folder = args.directory / 'model_evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    output = folder / f'attribution_{safe}_57facts_v2.csv'
    existing = v2.read_csv(output) if output.exists() else []
    by_sample = {}
    reference = {f['sample_id']: f for f in facts}
    for row in existing:
        sid = row['sample_id']
        if sid not in reference or sid in by_sample:
            raise ValueError(f'Unexpected or duplicated saved sample: {sid}')
        if row['model'] != model or row['prompt_version'] != v2.PROMPT_VERSION or row['reference_label'] != reference[sid]['fact_support_label']:
            raise ValueError(f'Existing result metadata mismatch for {sid}')
        by_sample[sid] = row
    schema = {'type':'object', 'properties': {'label': {'type':'string', 'enum':list(v2.LABELS)}, 'explanation': {'type':'string'}}, 'required':['label','explanation'], 'additionalProperties':False}
    print(f'\nModel: {model} | Prompt: {v2.PROMPT_VERSION} | Existing rows: {len(by_sample)}', flush=True)
    for i, f in enumerate(facts, 1):
        sid = f['sample_id']
        if sid in by_sample and by_sample[sid]['predicted_label'] in v2.LABELS and not by_sample[sid]['error']:
            print(f'[{i}/57] {sid}: already completed', flush=True)
            continue
        payload = {'model':model,'stream':False,'think':False,'format':schema,
                   'messages':[{'role':'system','content':v2.SYSTEM},
                               {'role':'user','content':v2.build_prompt(f, grouped[f['fact_id']])}],
                   'options':{'temperature':0.0,'num_predict':420},'keep_alive':'5m'}
        start = time.perf_counter()
        raw = ''
        response = {}
        label = ''
        explanation = ''
        error = ''
        flags = []
        try:
            response = v2.post_json(args.base_url.rstrip('/') + '/api/chat', payload, args.timeout)
            raw = response.get('message', {}).get('content', '')
            parsed = json.loads(raw)
            label = parsed.get('label', '')
            if label not in v2.LABELS:
                raise ValueError(f'Invalid label: {label!r}')
            explanation = str(parsed.get('explanation', '')).strip()
            if not explanation:
                flags.append('EMPTY_EXPLANATION')
            elif explanation[-1] not in '.!?':
                flags.append('POSSIBLY_TRUNCATED_EXPLANATION')
            if len(explanation) < 35:
                flags.append('VERY_SHORT_EXPLANATION')
            if response.get('done_reason') == 'length':
                flags.append('GENERATION_LENGTH_LIMIT')
        except (Exception) as exc:
            error = f'{type(exc).__name__}: {exc}'
            label = ''
            explanation = ''
        elapsed = round(time.perf_counter() - start, 2)
        by_sample[sid] = dict(sample_id=sid, fact_id=f['fact_id'], similarity_stratum=f['similarity_stratum'],
                              model=model, prompt_version=v2.PROMPT_VERSION, reference_label=f['fact_support_label'],
                              predicted_label=label, match=('YES' if label == f['fact_support_label'] else 'NO') if label else '',
                              explanation=explanation, raw_response=raw, error=error,
                              quality_flags=';'.join(flags), done_reason=response.get('done_reason',''), elapsed_seconds=elapsed)
        save(output, [by_sample[x['sample_id']] for x in facts if x['sample_id'] in by_sample])
        print(f'[{i}/57] {sid}: predicted={label or "ERROR"}, reference={f["fact_support_label"]}, elapsed={elapsed}s', flush=True)
        if error: print(f'  Error: {error}', flush=True)
        if flags: print(f'  Quality flags: {"; ".join(flags)}', flush=True)
    rows = [by_sample[f['sample_id']] for f in facts if f['sample_id'] in by_sample]
    valid = [r for r in rows if r['predicted_label'] in v2.LABELS and not r['error']]
    print(f'Saved: {output}\nValid: {len(valid)}/57 | Exact matches: {sum(r["match"] == "YES" for r in valid)}')
    print('Predicted labels:', dict(Counter(r['predicted_label'] for r in valid)))
    print('Quality flags:', dict(Counter(x for r in rows for x in r['quality_flags'].split(';') if x)))
    return len(valid) == 57


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='qwen3:1.7b', choices=MODELS)
    p.add_argument('--all-models', action='store_true', help='Run all six sequentially')
    p.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--timeout', type=int, default=300)
    args = p.parse_args()
    facts, grouped = load_inputs(args.directory)
    models = MODELS if args.all_models else (args.model,)
    ok = True
    for model in models:
        ok = run_model(model, args, facts, grouped) and ok
    if not ok:
        sys.exit(2)

if __name__ == '__main__':
    main()
