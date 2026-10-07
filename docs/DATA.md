# Data access and contract

The research dataset is not bundled with this code repository. Raw exports, reversible identity mappings, and private storage locations are excluded. Synthetic fixtures under `examples/sample_data/` exist only to exercise the code.

## Data layout

Prepare local dataset files using this logical layout:

```text
data/articles/cms_article.parquet
data/behaviors/{click,like,collect,comment,dislike}.parquet
splits/impressions_{train,val,test}.parquet
data/webrec_v1/interactions.csv
data/webrec_v1/item_features.csv
data/webrec_v1/{train,valid,test}.csv
data/webrec_v1/stats.csv
```

Keep checksums with local dataset files to verify their integrity.

## WebRec bridge contract

`scripts/run_integrated_pipeline.py` consumes:

- `train.csv`: `user_idx,item_idx,timestamp,interaction_type,weight`
- `item_features.csv`: `item_idx,title` plus optional `ai_synopsis`, `synopsis`, `language`, and `create_time`

The bridge fits the recommender only on `train.csv`, ranks unseen items, and writes:

- `articles.csv`: `id,title,ai_synopsis,synopsis,url,language`, where `id == item_idx`
- `daily_recommendations.csv`: `updated_at,user_id,article_ids`, where `user_id == user_idx`

The `url` field is blank when the source article schema does not provide one. Validation and test interactions are evaluation labels and are never substituted for model recommendations.

## Privacy guarantees

- Released `user_idx` and `item_idx` values are anonymous contiguous indices.
- The data preparation code never writes source database identifiers into release mapping tables.
- No public file should contain names, email addresses, phone numbers, credentials, production URLs, private paths, or external identity mappings.
- Do not attempt to re-identify users or join release identifiers to external sources.
- Generated prompts, model replies, logs, caches, and backtest outputs remain under ignored local directories.

Run both dataset validation and repository anonymity checks before a release:

```bash
python3 scripts/validate.py --data-root /path/to/release-data
python3 scripts/check_anonymity.py --root .
```
