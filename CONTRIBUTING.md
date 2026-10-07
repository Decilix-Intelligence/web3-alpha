# Contributing

Contributions should keep the benchmark reproducible.

## Pull Request Checklist

- Keep benchmark protocols explicit. Do not merge classic full-ranking results with LLM impression-ranking results.
- Add or update reproduction commands when adding a model.
- Write outputs under `benchmark/results/`.
- Avoid committing private credentials, raw database dumps, production logs, or large generated artifacts.
- Do not add names, personal email addresses, account handles, affiliations, local absolute paths, or reversible identity mappings.
- Run `python3 scripts/check_anonymity.py --root .` and `pytest` before submitting a change.
- Keep the `w3alpha` bridge fitted on `train.csv`; validation and test interactions are evaluation labels, not recommendation inputs.
- For new baselines, report at least `AUC`, `MRR`, and the relevant NDCG metrics.

## Adding a Classic-Track Model

1. Add the implementation or runner under `benchmark/`.
2. Use the split and metric definitions in `benchmark/evaluator.py`.
3. Append results to `benchmark/results/` with columns compatible with existing result files.

## Adding an LLM-Track Model

1. Use the Prompt4NR-compatible impression data.
2. Report results to `benchmark/results/llm_results.csv`.
3. Document the external implementation and model-weight setup because LLM-track results are sensitive to preprocessing and scoring details.
