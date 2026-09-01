"""Item-based collaborative filtering baseline."""

from __future__ import annotations

from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemCFRecommender:
    """Rank candidate items by similarity to the user's history."""

    def __init__(self, n_neighbors=30):
        self.name = "ItemCF"
        self.n_neighbors = n_neighbors

    def fit(self, train_df):
        self.users = list(train_df["user_idx"].unique())
        self.items = list(train_df["item_idx"].unique())
        self.user_idx_map = {u: i for i, u in enumerate(self.users)}
        self.item_idx_map = {i: j for j, i in enumerate(self.items)}
        self.idx_item_map = {j: i for i, j in self.item_idx_map.items()}

        rows = train_df["user_idx"].map(self.user_idx_map).to_numpy()
        cols = train_df["item_idx"].map(self.item_idx_map).to_numpy()
        data = train_df["weight"].to_numpy()
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(self.users), len(self.items))
        )
        self.item_similarity = cosine_similarity(self.user_item_matrix.T, dense_output=False)

    def recommend(self, user_id, user_history, candidate_items, n=None):
        if not user_history:
            ranked = list(candidate_items)
            return ranked if n is None else ranked[:n]

        item_scores = defaultdict(float)
        for hist_item in user_history:
            if hist_item not in self.item_idx_map:
                continue
            item_i = self.item_idx_map[hist_item]
            sim_scores = self.item_similarity[item_i].toarray().ravel()
            for idx, sim in enumerate(sim_scores):
                if sim <= 0:
                    continue
                candidate_item = self.idx_item_map[idx]
                if candidate_item in user_history:
                    continue
                item_scores[candidate_item] += float(sim)

        ranked = sorted(candidate_items, key=lambda item: (-item_scores.get(item, 0.0), item))
        return ranked if n is None else ranked[:n]
