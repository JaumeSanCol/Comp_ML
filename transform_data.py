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

    # Fit the pipeline
    transform_data_pipeline.fit(X_train, y_train)

    # Transform training data
    X_transformed = transform_data_pipeline.transform(X_train)

    # Load business dataset
    business = pd.read_csv("business_with_topics.csv")
    business_ids = business["business_id"].copy()

    # Transform business dataset
    business_features = transform_data_pipeline.transform(business)

    # ---- HERE: Build the correct column names ----

    # 1. Get the numeric feature names (those were scaled)
    numeric_features = NUM_FEATURES  # latitude, longitude, etc.

    # 2. Get the category columns (top categories + "Other")
    category_features = [
        f"cat_{cat.replace(' ', '_')}"
        for cat in transform_data_pipeline.named_steps["category_transformer"].top_categories
    ] + ["cat_Other"]

    # 3. Get the passthrough features from remainder
    passthrough_features = [
        col for col in X_train.columns
        if col not in transform_data_pipeline.named_steps["column_dropper"].columns_to_drop
        and col not in NUM_FEATURES
        and col != "categories"
    ]

    # 4. Final feature list
    final_columns = numeric_features + category_features + passthrough_features

    assert X_transformed.shape[1] == len(final_columns), "Mismatch between columns and features!"

    # ------------------------------------------------

    # Save everything with correct columns
    X_transformed_df = pd.DataFrame(X_transformed, columns=final_columns)
    business_features_df = pd.DataFrame(business_features, columns=final_columns)

    # Attach business_id
    business_transformed = pd.concat(
        [business_ids.reset_index(drop=True), business_features_df.reset_index(drop=True)],
        axis=1
    )

    # Save
    X_transformed_df.to_csv("training/X_transformed.csv", index=False)
    pd.DataFrame(y_train).to_csv("training/y_transformed.csv", index=False)
    business_transformed.to_csv("recommendation/business_transformed.csv", index=False)
