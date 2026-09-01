"""Popularity baseline."""

from __future__ import annotations


class PopularityRecommender:
    """Rank candidate items by global popularity."""

    def __init__(self):
        self.name = "Popularity"
        self.item_popularity = None

    def fit(self, train_df):
        self.item_popularity = (
            train_df.groupby("item_idx")["weight"].sum().sort_values(ascending=False)
        )

    def recommend(self, user_id, user_history, candidate_items, n=None):
        ranked = sorted(
            candidate_items, key=lambda item: (-self.item_popularity.get(item, 0.0), item)
        )
        return ranked if n is None else ranked[:n]
