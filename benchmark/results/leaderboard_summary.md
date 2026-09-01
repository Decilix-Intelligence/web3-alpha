# W3Alpha Benchmark Leaderboard

- Dataset: `webrec_v1`
- Effective evaluation users: 1655
- Classic-track rows merge local, RecBole, and text-modality full-ranking results.
- LLM-track rows are reported separately because they use impression-ranking candidate sets.

## Classic Track Ranking by AUC

| Rank | Model | Family | AUC | MRR | NDCG@10 | Recall@10 | Precision@10 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | LightGCN | RecBole | 0.875143 | 0.098486 | 0.115133 | 0.186103 | 0.018610 |
| 2 | BPR | RecBole | 0.871191 | 0.107221 | 0.125038 | 0.201208 | 0.020121 |
| 3 | ItemCF | local baseline | 0.859776 | 0.171692 | 0.211407 | 0.355891 | 0.035589 |
| 4 | NCF | RecBole | 0.827560 | 0.104976 | 0.122685 | 0.203021 | 0.020302 |
| 5 | TextMoRec-BERTchinese-SASRec-seq100 | text modality | 0.801508 | 0.145950 | 0.181553 | 0.297281 | 0.029728 |
| 6 | UserCF | local baseline | 0.792908 | 0.153463 | 0.182308 | 0.294864 | 0.029486 |
| 7 | SASRec | RecBole | 0.751115 | 0.078816 | 0.096715 | 0.158308 | 0.015831 |
| 8 | GRU4Rec | RecBole | 0.650997 | 0.071375 | 0.087558 | 0.138973 | 0.013897 |
| 9 | Popularity | local baseline | 0.627406 | 0.119572 | 0.151546 | 0.250755 | 0.025076 |

## LLM Track Results

| Model | AUC | MRR | nDCG@5 | nDCG@10 | Test impressions | Status |
|---|---:|---:|---:|---:|---:|---|
| Prompt4NR | 0.8285 | 0.6383 | 0.6438 | 0.6815 | 43840 | completed |
