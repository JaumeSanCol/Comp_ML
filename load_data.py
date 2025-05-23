import json
import os
import pickle

import pandas as pd


def load_data() -> dict[str, pd.DataFrame]:
    """Load all the data and return it as DataFrames inside a dictionary.

    This should only be used if you want all data, as it takes a lot of time to load
    (the size of the dataset is big). If you only need data for a single city,
    which will be the case after selecting a city, use `load_city_data` instead.
    """
    data_path = "data"
    reviews_path = "yelp_academic_dataset_review.json"
    business_path = "yelp_academic_dataset_business.json"

    with open(os.path.join(data_path, reviews_path), "r", encoding="utf-8") as f:
        review_data = [json.loads(line) for line in f]

    with open(os.path.join(data_path, business_path), "r", encoding="utf-8") as f:
        business_data = [json.loads(line) for line in f]

    review_df = pd.DataFrame(review_data)
    business_df = pd.DataFrame(business_data)

    return {"reviews": review_df, "businesses": business_df}


# noinspection PyTypeChecker
def load_city_data(city: str) -> pd.DataFrame:
    """Load only the data pertaining to one city.

    Is much faster than calling `load_data` and filtering the results, as this function already does that,
    and caches the results for later calls.
    """
    file_path = os.path.join("data", "city_data.pkl")
    if not os.path.exists(file_path):
        data = load_data()
        review_df = data["reviews"]
        business_df = data["businesses"]

        businesses_in_city = business_df[business_df["city"] == city].reset_index(
            drop=True
        )
        business_ids = businesses_in_city["business_id"]

        # before reindexing reviews, sort them by the primary key
        # this is to ensure the same ordering no matter what
        # and to have the same train-test split indices all the time.
        reviews_in_city = (
            review_df[review_df["business_id"].isin(business_ids)]
            .sort_values(by="review_id")
            .reset_index(drop=True)
        )

        # we will assume a business is a gastronomy business if it has "Restaurants" in its categories.
        restaurant_df = businesses_in_city[
            businesses_in_city["categories"].str.contains("Restaurants", na=False)
        ].reset_index(drop=True)

        city_data = {
            "reviews": reviews_in_city,
            "businesses": businesses_in_city,
            "restaurants": restaurant_df,
        }

        with open(file_path, "wb") as f:
            pickle.dump(city_data, f)
    else:
        with open(file_path, "rb") as f:
            city_data = pickle.load(f)
    return city_data
