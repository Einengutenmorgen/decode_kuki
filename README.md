# kuki-decode: can a model decode the insider meaning of a coded phrase?
Standalone from the S1-S5 detection pipeline. Input: `data/kuki_{ru,tr}_aggregate.jsonl` (or set `KUKI_DATA`).

1. `python build_datasets.py` - one dataset per condition in `datasets/` (inputs + rendered chat messages), plus `datasets/_references` (literal/insider/why per case).
2. `python tests/test_build.py` - sanity checks (set `OLD_SPANS` to compare paragraph indices with the old pipeline).
3. `vllm serve <hf_id>`, then `python run_condition.py --model <alias> --condition <id> --split dev` (aliases in `config.MODELS`; `--dry-run` tests the plumbing).
4. `judge.py` - symmetric LLM judge: two explanations of a coded phrase (A/B, order balanced per split and language) -> equivalent / partially_equivalent / not_equivalent. Blind to condition and model; the judge model must not be one of the evaluated models. Rubric version `j2`.
   - dev: `for r in results/*__v2.jsonl; do python judge.py run --results $r --judge-model <m> --base-url <url> --split dev; done` (`*__unnamed` conditions are skipped, see TODO)
   - controls: `python judge.py control --kind literal|shuffled --split dev --judge-model <m> --base-url <url>`
   - validation: `python judge.py sample --judgments judgments/ctx-*__j2.jsonl --n 100`, fill `human_verdict` in `judgments/label_sheet.csv`, then `python judge.py agree`
   - test only after the agreement is accepted. `--runs 2` also judges repeated generations (run-to-run noise).
   - TODO: `score_unnamed.py` (standalone) for the unnamed detection-ablation conditions; not written.
   

Rules: freeze prompts and the judge rubric on `dev` only; the variance set (`is_multiref`) is run last.
Conditions: one factor at a time around the default (paragraph, no title, no outlet, English, phrase named); see `config.CONDITIONS`.
