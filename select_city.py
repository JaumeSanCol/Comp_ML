#%% md
# # Exploring the cities
#
# The goal of this notebook is to explore the businesses and reviews in different cities, to choose a city for which we will create a recommendation system.
#%%
import json
import os

import pandas as pd

DATA_PATH = "data"
REVIEWS_PATH = "yelp_academic_dataset_review.json"
BUSINESS_PATH = "yelp_academic_dataset_business.json"

with open(os.path.join(DATA_PATH, REVIEWS_PATH), "r", encoding="utf-8") as f:
    review_data = [json.loads(line) for line in f]

with open(os.path.join(DATA_PATH, BUSINESS_PATH), "r", encoding="utf-8") as f:
    business_data = [json.loads(line) for line in f]

review_df = pd.DataFrame(review_data)
business_df = pd.DataFrame(business_data)
#%%
del review_data
del business_data
#%%
review_df
#%%
business_df
#%%
print(f"There are {len(business_df)} businesses and {len(review_df)} reviews.")
business_features = [col for col in business_df.columns if not col.endswith("_id")]
review_features = [col for col in review_df.columns if not col.endswith("_id")]
print(f"Business data has {len(business_features)} features. Their names are {business_features}.")
print(f"Review data has {len(review_features)} features. Their names are {review_features}.")
#%% md
# Now let us choose one of the cities for our analysis and recommendation system. The reason for limiting ourselves to one city is simply because the dataset is too big; the computers we are working with do not have enough memory to fit all the data in it.
# 
# The second reason is that it makes sense to make recommendations to the user based on the city they are currently in (if we are vising Porto, we don't want recommendations for businesses in Lisbon). We could make a big recommender system that tries to generalize across the cites, or we can make smaller and specialized models, one for each city of interest. We have chosen the smaller and specialized model approach.
#%%
from collections import Counter

cities_count = Counter(business_df["city"])
cities_with_most_businesses = [city for city, count in cities_count.most_common(100)]
#%%
data = []

for city in cities_with_most_businesses:
    business_in_city_df = business_df[business_df["city"] == city]
    num_businesses = business_in_city_df.shape[0]

    business_ids = set(business_in_city_df["business_id"])
    reviews_in_city_df = review_df[review_df["business_id"].isin(business_ids)]

    num_reviews = reviews_in_city_df.shape[0]
    stddev = reviews_in_city_df["stars"].std()
    mean = reviews_in_city_df["stars"].mean()

    data.append({
        "city": city,
        "num_businesses": num_businesses,
        "num_reviews": num_reviews,
        "stddev": stddev,
        "mean": mean
    })

df = pd.DataFrame(data)
df
#%% md
# After inspecting the data manually for a little bit (PyCharm has great tooling for exploring the dataframe), we chose Sparks as the city we are going to make recommendations for based on the following factors:
# 
# 1. The target variable `stars` has one of the highest standard deviation among all the cities with the value of `1.62`. This means there is a greater variety in the ratings, making it possible to have a more varied recommendation system.
# 2. The mean of `stars` is `3.61`, meaning there are businesses which clients like, and which they don't like (it isn't one sided with businesses only having high rating), which should allow us to recommend businesses that are more suited to the user, not just popular in general.
# 3. There are a sufficient number of businesses, standing at `1624`, and a good amount of reviews, namely `73033`. This gives us sufficient data to work with.
# 4. The name `Sparks` is cool.
#%%
city = "Sparks"