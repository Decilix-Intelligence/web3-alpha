"""User-based collaborative filtering baseline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserCFRecommender:
    """Rank candidate items by neighbor-user similarity."""

    def __init__(self, n_neighbors=30):
        self.name = "UserCF"
        self.n_neighbors = n_neighbors

    def fit(self, train_df):
        self.users = list(train_df["user_idx"].unique())
        self.items = list(train_df["item_idx"].unique())
        self.user_idx_map = {u: i for i, u in enumerate(self.users)}
        self.item_idx_map = {i: j for j, i in enumerate(self.items)}

        rows = train_df["user_idx"].map(self.user_idx_map).to_numpy()
        cols = train_df["item_idx"].map(self.item_idx_map).to_numpy()
        data = train_df["weight"].to_numpy()
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(self.users), len(self.items))
        )
        self.user_similarity = cosine_similarity(self.user_item_matrix, dense_output=False)
        self.user_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def recommend(self, user_id, user_history, candidate_items, n=None):
        if user_id not in self.user_idx_map:
            ranked = list(candidate_items)
            return ranked if n is None else ranked[:n]

        user_i = self.user_idx_map[user_id]
        sim_scores = self.user_similarity[user_i].toarray().ravel()
        neighbor_idx = np.argsort(sim_scores)[::-1][1 : self.n_neighbors + 1]

        item_scores = defaultdict(float)
        for sim_user_i in neighbor_idx:
            sim = float(sim_scores[sim_user_i])
            if sim <= 0:
                continue
            sim_user = self.users[sim_user_i]
            for item in self.user_items.get(sim_user, ()):
                if item in user_history:
                    continue
                item_scores[item] += sim

        ranked = sorted(candidate_items, key=lambda item: (-item_scores.get(item, 0.0), item))
        return ranked if n is None else ranked[:n]
