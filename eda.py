#%%
from load_data import load_city_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

#%%
data = load_city_data("Philadelphia")
business_df = data["businesses"]
reviews_df = data["reviews"]
#%% md
# # Reviews data
#%%
reviews_df.info()
#%%
reviews_df.describe()
#%% md
# # Business data
#%%
business_df.info()
#%%
business_df.describe()
#%% md
# Let's look at the correlation between the numerical attributes.
#%%
business_df[["latitude", "longitude", "stars", "review_count"]].corr()
#%% md
# Virtually none, which means the numerical features for the businesses are linearly independent. We will later also look at the correlation with the target variable after joining the two dataframes.
#%%
_, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].boxplot(business_df["stars"])
ax[0].set_title("Boxplot of `stars` for Businesses")
ax[0].set_xlabel("stars")
ax[0].set_ylabel("Count")

ax[1].boxplot(business_df["review_count"])
ax[1].set_title("Boxplot of `review_count` for Businesses")
ax[1].set_xlabel("review_count")
ax[1].set_ylabel("Count")

ax[2].hist(business_df["is_open"], bins=2, edgecolor='black')
ax[2].set_title("Histogram of `is_open` for Businesses")
ax[2].set_xlabel("is_open")
ax[2].set_ylabel("Count")
ax[2].set_xticks([0, 1], ['Closed', 'Open'])

plt.tight_layout()
plt.show()

#%% md
# ## Making sense of `categories`
#%%
business_df["categories"] = business_df["categories"].str.split(", ")
#%%
flattened_categories = [item for sublist in business_df["categories"].dropna() for item in sublist]

# how many different categories are there?
print(f"There are {len(set(flattened_categories))} unique categories.")

# what are their counts?
counts = Counter(flattened_categories)
print("The most common categories are:")
print(counts.most_common(50))
#%%
import matplotlib.pyplot as plt

accumulative_categories_counts = [sum(count for _, count in counts.most_common(i)) for i in range(1, len(counts) + 1)]

# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 2 columns

# Full accumulative plot
ax[0].plot(accumulative_categories_counts)
ax[0].set_title("Accumulated Category Counts")
ax[0].set_xlabel("Number of Categories")
ax[0].set_ylabel("Total Count")

# First 100 categories plot
ax[1].plot(accumulative_categories_counts[:100])
ax[1].set_title("Accumulated Counts (Top 100 Categories)")
ax[1].set_xlabel("Number of Categories")
ax[1].set_ylabel("Total Count")

# First 50 categories plot
ax[2].plot(accumulative_categories_counts[:50])
ax[2].set_title("Accumulated Counts (Top 50 Categories)")
ax[2].set_xlabel("Number of Categories")
ax[2].set_ylabel("Total Count")

# Adjust layout and show
plt.tight_layout()
plt.show()
#%% md
# We will limit ourselves only to gastronomy related establishments.
#%%
business_df["categories"] = business_df["categories"].apply(lambda x: ", ".join(x) if x is not None else "")
#%%
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-MiniLM-L6-v2')
anchor = "food, restaurant, eating, drinking"
anchor_embedding = embedder.encode(anchor, convert_to_numpy=True)
business_categories_embeddings = embedder.encode(business_df["categories"].values, convert_to_numpy=True)
similarities = util.cos_sim(anchor_embedding, business_categories_embeddings).numpy()
#%%
threshold = 0.6
mask = (similarities >= threshold).flatten()
restaurant_df = business_df[mask]
restaurant_df.to_csv('philly.csv', index=False, encoding='utf-8')
#%%
print(f'There are {reviews_df["business_id"].isin(set(restaurant_df["business_id"])).to_numpy().sum()} reviews for restaurants.')
#%% md
# # #TODO: make further analysis on `restaurant_df` instead of `business_df`
#%% md
# Now let us explore if there are any businesses that have a very low number of reviews. If there are, we may want to discard them, as a lack of reviews may introduce undesired noise and lead to our model performing worse.
#%%
max_x = 20
thresholds = list(range(max_x))
counts = [len(restaurant_df[restaurant_df["review_count"] <= x]) for x in thresholds]

plt.figure(figsize=(15, 5))
plt.bar(thresholds, counts, edgecolor='black')
plt.title("Number of restaurants with review_count <= x")
plt.xlabel("Review Count (x)")
plt.ylabel("Number of restaurants with review_count <= x")
plt.xticks(thresholds)
plt.tight_layout()
plt.show()

#%% md
# # Combined data
#%%
full_df = pd.merge(business_df, reviews_df, how="left", on="business_id")
full_df.rename(columns={"stars_x": "business_stars", "stars_y": "stars"}, inplace=True)
full_df.info()
#%%
full_df.drop(["business_id", "name", "address", "city", "postal_code", "attributes", "categories", "hours", "user_id", "useful", "funny", "cool", "date"], axis=1, inplace=True)
#%%
full_df[["latitude", "longitude", "business_stars", "review_count", "stars"]].corr()