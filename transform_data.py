from collections import Counter

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def _split_categories(value):
    if isinstance(value, str):
        return value.split(", ")
    if isinstance(value, list):
        return value
    # value is NaN or something else
    return []


# noinspection PyPep8Naming
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns_to_drop: list[str] | None = None,
    ):
        if columns_to_drop is None:
            columns_to_drop = [
                "business_id",
                "name",
                "address",
                "city",
                "state",
                "postal_code",
                "categories",
                "is_open",
                "attributes",
                "hours",
                "review_id",
                "user_id",
                "useful",
                "funny",
                "cool",
                "text",
                "date",
            ]
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(self.columns_to_drop, axis=1)


# noinspection PyPep8Naming
class CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 20):
        self.top_n = top_n
        self.top_categories = None

    def fit(self, X, y=None):
        # Extract and store top categories during fit
        df = X.copy()
        df["categories"] = df["categories"].apply(_split_categories)

        flattened_categories = [
            item for sublist in df["categories"] for item in sublist if item != ""
        ]

        # Select 2 categories more in case "Restaurants" or empty string appear
        # Note: all categories will have "Restaurants" as a category, hence the removal
        top_categories_counts = Counter(flattened_categories).most_common(
            self.top_n + 2
        )
        top_categories, _ = zip(*top_categories_counts)

        # Remove "Restaurants" and empty strings if they appear
        self.top_categories = [
            cat for cat in top_categories if cat != "Restaurants" and cat != ""
        ][: self.top_n]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["categories"] = df["categories"].apply(_split_categories)

        # Create binary features for each top category
        for category in self.top_categories:
            df[f"cat_{category.replace(' ', '_')}"] = df["categories"].apply(
                lambda x: 1 if category in x else 0
            )

        # Add "Other" category feature for any restaurant with categories not in top_n
        df["cat_Other"] = df["categories"].apply(
            lambda x: (
                1
                if any(
                    cat not in self.top_categories
                    and cat != "Restaurants"
                    and cat != ""
                    for cat in x
                )
                else 0
            )
        )

        return df


NUM_FEATURES = ["latitude", "longitude", "restaurant_stars", "review_count"]

transform_data_pipeline = Pipeline(
    [
        ("category_transformer", CategoryTransformer()),
        ("column_dropper", ColumnDropper()),
        (
            "num_scaling",
            ColumnTransformer(
                [("standard_scaler", StandardScaler(), NUM_FEATURES)],
                remainder="passthrough",
            ),
        ),
    ]
)

if __name__ == "__main__":
    import numpy as np
    from split_data import split_data

    X_train, X_test, y_train, y_test = split_data()
    transform_data_pipeline.fit_transform(X_train, y_train)

    X_transformed = transform_data_pipeline.transform(X_train)
    assert not np.any(np.isnan(X_transformed)), "NaNs in transformed data"
        # Guardar en un CSV
    df_transformed = pd.DataFrame(X_transformed)
    y_train = pd.DataFrame(y_train)
    print(len(X_train))
    print(len(y_train))
    print(len(X_transformed))
    df_transformed.to_csv("training/X_transformed.csv", index=False)
    y_train.to_csv("training/y_transformed.csv", index=False)
