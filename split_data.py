import pandas as pd
from sklearn.model_selection import train_test_split

from load_data import load_city_data


# noinspection PyShadowingNames
def split_data():
    data = load_city_data(city="Philadelphia")
    review_df = data["reviews"]
    business_df = data["businesses"]
    restaurant_df = business_df[
        business_df["categories"].str.contains("Restaurants", na=False)
    ]

    full_df = (
        pd.merge(restaurant_df, review_df, how="left", on="business_id")
        .sort_values(by="review_id")
        .reset_index(drop=True)
    )
    full_df.rename(
        columns={"stars_x": "restaurant_stars", "stars_y": "stars"}, inplace=True
    )

    y = full_df["stars"].values
    full_df.drop(columns=["stars"], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        full_df, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()
