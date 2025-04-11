# %%
import pandas as pd
import json
import numpy as np
import os

# %% [markdown]
# ## Prepare

# %%
philly=pd.read_csv('philly.csv', encoding='utf-8')

# %%
data_path = "data"
reviews_path = "yelp_academic_dataset_review.json"

with open(os.path.join(data_path, reviews_path), "r", encoding="utf-8") as f:
    review_data = [json.loads(line) for line in f]
reviews = pd.DataFrame(review_data)

# %%
philly.columns

# %%
reviews.columns

# %%
# Merge both DataFrames on 'business_id', keeping only matching reviews
filtered_merge= reviews.merge(philly, on="business_id", how="inner",suffixes=("_review", "_business"))

# %%
filtered_reviews=filtered_merge.drop(columns=['hours','useful', 'funny', 'cool','latitude', 'longitude','name', 'address', 'city', 'state','postal_code','is_open','attributes','categories',])

# %%
filtered_reviews.describe()

# %%
#filtered_reviews=filtered_reviews[filtered_reviews['review_count']>=70]

# %%
filtered_reviews

# %% [markdown]
# ### Remove users/business with low number of reviews

# %%
# Sort by date (most recent first) and keep only the latest review per user-business pair
remove_duplicates= filtered_reviews.sort_values(by="date", ascending=False).drop_duplicates(subset=["business_id", "user_id"], keep="first")

# Count the number of reviews per user
user_review_counts = remove_duplicates["user_id"].value_counts()

# Keep only users with at least 10 reviews
valid_users = user_review_counts[user_review_counts >= 10].index

# Filter the DataFrame
corpus_filtered = remove_duplicates[remove_duplicates["user_id"].isin(valid_users)]

# %%
corpus_filtered

# %% [markdown]
# ### Lowecasing

# %%
review_lower=corpus_filtered
review_lower["text"]=review_lower["text"].str.lower()
review_lower

# %%
corpus=review_lower

# %% [markdown]
# ### First Cleaning
# 

# %%
import string
import re

# Remove newlines, ellipses, and numbers
corpus["text"] = corpus["text"].apply(lambda text: text.replace("\n", "").replace("...", ""))

# Remove punctuation and numbers
corpus["text"] = corpus["text"].apply(lambda text: ''.join([char for char in text if char not in string.punctuation and not char.isdigit()]))

# Alternatively, you can use a regular expression to remove numbers
corpus["text"] = corpus["text"].apply(lambda text: re.sub(r'\d+', '', text))



# %%
corpus

# %% [markdown]
# ### Lemmatization

# %%
import spacy

nlp = spacy.load("en_core_web_sm")


# %%
corpus['words'] = None
for index, sample in corpus.iterrows():
    list_words = [] 
    sent = nlp(sample.text)
    
    for token in sent:
        list_words.append(token.lemma_)
    
    corpus.at[index, "words"] = list_words 


# %%
corpus

# %% [markdown]
# ### Stopwords

# %%
import nltk
import ast  # Para convertir strings en listas
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

corpus["words"] = corpus["words"].apply(lambda x: [word for word in x if word.lower() not in stop_words and not word.isdigit()])


# %%
corpus

# %% [markdown]
# ### Only english

# %%
import nltk
import pandas as pd

# Download the words corpus from nltk
nltk.download('words')
from nltk.corpus import words

# Create a set of English words (lowercased for case-insensitive checking)
english_words_set = set(words.words())

# Function to filter English words using the predefined English word list
def filter_english_words(word_list):
    filtered_words = [word for word in word_list if word.lower() in english_words_set]
    return filtered_words


corpus["words"] = corpus["words"].apply(filter_english_words)

print(corpus)


# %%
# Remove empty lists
corpus = corpus[corpus["words"].apply(lambda x: len(x) > 0)]


# %%
corpus

# %% [markdown]
# ### Common words

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Aplanar la lista de listas y unir palabras en un solo string
flat_words = [word for sublist in corpus["words"] for word in sublist]
text = " ".join(flat_words)

# Crear el WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)

# Graficar el WordCloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Ocultar ejes
plt.title("Word Cloud de Corpus")
plt.show()


# %%
corpus["words"] = corpus["words"].apply(lambda sublist: [word for word in sublist if word.strip()])

# %%
corpus['words_join'] = None
corpus.loc[:, "words_join"] = corpus["words"].apply(lambda x: " ".join([word for word in x if word]))

# %%
corpus

# %%
corpus=corpus.drop(columns=["date","review_count","stars_business"])

# %% [markdown]
# ### Save cleaned corpus

# %%
corpus.to_csv('lemma.csv', index=False, encoding='utf-8')

# %% [markdown]
# ## READ

# %%
import pandas as pd
corpus = pd.read_csv('lemma.csv', encoding='utf-8')

# %% [markdown]
# ### VADER POLARITY

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus["text"])

print(X.shape)

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# %%
y_pred = []
for rev in corpus["text"]:
    y_pred.append(1 if analyzer.polarity_scores(rev)['compound'] > 0 else 0)

# %%
corpus["vader"]= y_pred
corpus["text"][corpus["vader"]==0]

# %%
corpus

# %% [markdown]
# ### adjectives only

# %%
import spacy
import pandas as pd
from tqdm import tqdm  # For progress bar

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to extract only adjectives from a text
def extract_adjectives(text):
    doc = nlp(text)  # Process the full text
    return [token.text for token in doc if token.pos_ == "ADJ"]

tqdm.pandas(desc="Extracting adjectives...")

# Apply the function with progress bar
corpus['adj'] = corpus['words'].progress_apply(extract_adjectives)

# Print the result
print(corpus)


# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Aplanar la lista de listas y unir palabras en un solo string
flat_words = [word for sublist in corpus["adj"] for word in sublist]
text = " ".join(flat_words)

# Crear el WordCloud
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)

# Graficar el WordCloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Ocultar ejes
plt.title("Word Cloud de Corpus")
plt.show()


# %%
irrelevant_words=["good"]
corpus["adj"] = corpus["adj"].apply(lambda sublist: [word for word in sublist if word not in irrelevant_words])

# %%
corpus['adj_join'] = None
corpus.loc[:, "adj_join"] = corpus["adj"].apply(lambda x: " ".join([word for word in x if word]))

# %%
corpus

# %% [markdown]
# ## LDA

# %%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
num_batches = 7
batches = np.array_split(corpus, num_batches)


# %%
from collections import Counter

# Define helper functions
def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


# %%
num_batches

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

X = vectorizer.fit_transform(corpus["adj_join"])

# Try different numbers of topics and record scores
topic_range = range(2, 15)  # Test from 2 to 15 topics
perplexities = []
log_likelihoods = []

for n_topics in topic_range:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    perplexities.append(lda.perplexity(X))  # Lower is better
    log_likelihoods.append(lda.score(X))   # Higher is better

# Plot results
fig, ax1 = plt.subplots()

ax1.plot(topic_range, perplexities, marker="o", color="red", label="Perplexity")
ax1.set_xlabel("Number of Topics")
ax1.set_ylabel("Perplexity (lower is better)", color="red")

ax2 = ax1.twinx()
ax2.plot(topic_range, log_likelihoods, marker="s", color="blue", label="Log-Likelihood")
ax2.set_ylabel("Log-Likelihood (higher is better)", color="blue")

plt.title("Choosing the Optimal Number of Topics")
plt.show()

# %%
corpus

# %%

n_topics = 6
num_batches=1
batches = np.array_split(corpus, num_batches)


# %%
n_topics

# %%
# Define helper functions
def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        print(temp_vector_sum)
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
        
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word)  # Simply store the word as-is
        top_words.append(" ".join(topic_words))         
    return top_words

# %%
# Store topic distributions
all_topic_matrices = []

for batch_idx, batch in enumerate(batches):
    print(f"\nProcessing Batch {batch_idx+1}/{num_batches}...\n")
    
    # Feature extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus["adj_join"].to_numpy())

    # LDA model
    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0)
    lda_topic_matrix = lda_model.fit_transform(X)

    # Store the topic distributions
    all_topic_matrices.append(pd.DataFrame(lda_topic_matrix))

    # Get topic distribution (for visualization)
    lda_keys = get_keys(lda_topic_matrix)
    lda_categories, lda_counts = keys_to_counts(lda_keys)
    top_n_words_lda = get_top_n_words(20, lda_keys, X, vectorizer)

    # Print results
    for i in range(len(top_n_words_lda)):
        print(f"Topic {i+1}: {top_n_words_lda[i]}")

# Combine all topic matrices
df_topics = pd.concat(all_topic_matrices, ignore_index=True)


# %%
for i in range(len(top_n_words_lda)):
    print("Topic {}: ".format(i+1), top_n_words_lda[i])

# %%
from collections import Counter

# Asegurar que cada tópico es una lista de palabras
top_n_words_lda = [topic.split() for topic in top_n_words_lda]  # <- Agregar esta línea si necesario

# Contar palabras en todos los tópicos
word_counts = Counter(word for topic in top_n_words_lda for word in topic)

# Filtrar palabras que aparecen en más de un tópico
repeated_words = {word for word, count in word_counts.items() if count > 4}

# Crear una copia de top_n_words_lda sin palabras repetidas
filtered_top_n_words_lda = [
    [word for word in topic if word not in repeated_words]
    for topic in top_n_words_lda
]

# Mostrar palabras repetidas
print("Palabras repetidas en múltiples tópicos:", repeated_words)

# Mostrar los nuevos tópicos sin palabras repetidas
for i, topic in enumerate(filtered_top_n_words_lda, 1):
    print(f"Tópico {i}:", topic)


# %% [markdown]
# ## Visualization

# %%
# add the topics to this dataset as 8 new columns
df_topics.columns = ['topic_{}'.format(i) for i in range(n_topics)]

# Combine with the original dataset
dataset_f = pd.concat([corpus, df_topics], axis=1)
dataset_f.head()


# %%
dataset_f.to_csv('topics.csv', index=False, encoding='utf-8')

# %%
type(corpus["words"])


# %%
# Convertir a lista de listas
sentences = corpus["words"].tolist()
type(sentences)

# %%
def preprocess(sentence):
    # Remove punctuation
    sentence = ''.join([char for char in sentence if char not in string.punctuation])
    # Split into words
    words = sentence.split()
    return words

# Apply the preprocessing to your sentences
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]


# %%
from gensim.models import Word2Vec

model = Word2Vec(preprocessed_sentences, 
                 vector_size=100,  # Dimensión de los embeddings
                 window=5,         # Contexto de palabras
                 min_count=2,      # Ignora palabras con menos de 2 apariciones
                 workers=4,        # Paralelización
                 sg=1)             # Skip-gram (sg=1) o CBOW (sg=0)

print(list(model.wv.index_to_key))  # Muestra algunas palabras del vocabulario


# %%
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random

def reduce_dimensions(model, num_dimensions=2, words=[]):
    vectors = []
    labels = []
    
    if not words:
        words = model.wv.index_to_key

    for word in words:
        vectors.append(model.wv[word])
        labels.append(word)

    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    return vectors, labels

def plot_words_by_topic(vectors, labels, topics):
    plt.figure(figsize=(10, 8))

    # Crear una paleta de colores con suficientes colores únicos
    num_topics = len(topics)
    cmap = plt.cm.get_cmap("tab20", num_topics)  # Hasta 20 colores únicos

    topic_colors = {i: cmap(i) for i in range(num_topics)}

    word_to_topic = {}  # Diccionario para mapear palabra -> tópico
    for topic_idx, words in enumerate(topics):
        for word in words:
            word_to_topic[word] = topic_idx

    for i, label in enumerate(labels):
        x, y = vectors[i]
        topic_idx = word_to_topic.get(label, -1)
        color = topic_colors.get(topic_idx, "gray")  # Gris si el tópico no se encuentra
        plt.scatter(x, y, color=color, alpha=0.7)
        plt.text(x + 0.1, y + 0.1, label, fontsize=9, color=color)

    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.title("Words-topics t-SNE")
    plt.show()



# %%
print(list(model.wv.index_to_key)[:20])  # Muestra algunas palabras del vocabulario


# %%
# Extraer palabras únicas de filtered_top_n_words_lda
words_to_plot = list(set(word for topic in filtered_top_n_words_lda for word in topic))
words_to_plot =[char for char in words_to_plot  if char != "green"]

# Reducir dimensionalidad
vectors_2d, labels = reduce_dimensions(model, num_dimensions=2, words=words_to_plot)

# Graficar con colores por tópico
plot_words_by_topic(vectors_2d, labels, filtered_top_n_words_lda)

# %% [markdown]
# ## Agregate

# %%
dataset_f = pd.read_csv('topics.csv', encoding='utf-8')

# %%
dataset_f.columns

# %% [markdown]
# ### Useful

# %%
reviews.columns

# %%
useful=reviews.drop(columns=[ 'user_id', 'business_id', 'stars', 'funny','cool', 'text', 'date'])

# %%
dataset_f=dataset_f.merge(useful, on="review_id", how="inner",)

# %%
dataset_f.columns

# %%
import pandas as pd

# Assuming dataset_f is your DataFrame
# Function to adjust the topic values based on vader
def adjust_values(row):
    if row['vader'] == 0:
        # If vader is 0, multiply topic values by -1 to make them subtract
        return row[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']] * -1
    return row[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']]

# Apply the function to adjust the topic values based on vader
dataset_f[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']] = dataset_f.apply(adjust_values, axis=1)

# Now group by 'user_id' and sum the topic values
results = dataset_f.groupby('user_id')[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']].sum()

# Multiply each topic column by the 'useful' column for each row
dataset_f_weighted = dataset_f[['business_id', 'useful', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']].copy()

# Apply the weighting (multiplying the topic columns by the 'useful' column)
for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']:
    dataset_f_weighted[topic] = dataset_f_weighted[topic] * dataset_f_weighted['useful']

# Now, group by 'business_id' and sum the weighted topic columns
business_results = dataset_f_weighted.groupby('business_id')[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']].sum()


merged_df = dataset_f.merge(results, on='user_id', suffixes=('_user', '_business'))

# Subtract the topic values for each review
for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']:
    merged_df[topic] = merged_df[f'{topic}_user'] - merged_df[f'{topic}_business']

# Drop the intermediate columns used for the merge (optional, for clarity)
merged_df.drop(columns=[f'{topic}_user' for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']] +
                  [f'{topic}_business' for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']], inplace=True)

# Display the final DataFrame
merged_df


# %%
import pandas as pd

# Multiply each topic column by the 'useful' column for each row
dataset_f_weighted = dataset_f[['business_id', 'useful', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']].copy()

# Apply the weighting (multiplying the topic columns by the 'useful' column)
for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']:
    dataset_f_weighted[topic] = dataset_f_weighted[topic] * dataset_f_weighted['useful']

# Now, group by 'business_id' and sum the weighted topic columns
business_results = dataset_f_weighted.groupby('business_id')[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']].sum()


merged_df = dataset_f.merge(results, on='user_id', suffixes=('_user', '_business'))

# Subtract the topic values for each review
for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']:
    merged_df[topic] = merged_df[f'{topic}_user'] - merged_df[f'{topic}_business']

# Drop the intermediate columns used for the merge (optional, for clarity)
merged_df.drop(columns=[f'{topic}_user' for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']] +
                  [f'{topic}_business' for topic in ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']], inplace=True)

# Display the final DataFrame
merged_df


# %%
merged_df.columns

# %%
traindataset=merged_df.drop(columns=['review_id', 'user_id', 'business_id','text', 'words', 'words_join','adj', 'adj_join', ])

# %%
traindataset=traindataset.rename(columns={'stars_review': 'y'}, inplace=False)


# %% [markdown]
# ## TRAINING

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Assuming 'traindataset' is your DataFrame
# Step 1: Split into X and y
X = traindataset[['vader','topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5']]
y = traindataset['y']

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the features (X_train and X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 4: Train a Random Forest Regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_regressor.predict(X_test_scaled)

# Step 6: Evaluate the model (for example, using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Step 3: Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# %%
import numpy as np
import matplotlib.pyplot as plt

# Apply jitter to predicted and actual values to reduce overlap
jitter_strength = 0.15  # Adjust this to increase/decrease jitter
y_test_jittered = y_test + np.random.uniform(-jitter_strength, jitter_strength, len(y_test))
y_pred_jittered = y_pred + np.random.uniform(-jitter_strength, jitter_strength, len(y_pred))

# Plot predicted vs actual values with alpha transparency and jitter
plt.figure(figsize=(8, 6))
plt.scatter(y_test_jittered, y_pred_jittered, alpha=0.05, color='blue')  # Apply alpha transparency
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values with Jitter and Transparency')

# Show plot
plt.show()


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Round the predicted values to make them discrete
y_pred_rounded = y_pred.round()

# Step 2: Confusion Matrix (after rounding)
cm = confusion_matrix(y_test, y_pred_rounded)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(int(min(y_test)), int(max(y_test)) + 1)],
            yticklabels=[str(i) for i in range(int(min(y_test)), int(max(y_test)) + 1)])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (After Rounding)')
plt.show()





