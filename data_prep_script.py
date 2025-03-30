# %%
import pandas as pd
import json
import numpy as np


# %% [markdown]
# ## Prepare

# %%
checkin_path="/home/jaume/Escritorio/Ultim_Semestre/Comp.ML/Practicas/project/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_checkin.json"
business_path="/home/jaume/Escritorio/Ultim_Semestre/Comp.ML/Practicas/project/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json"
review_path="/home/jaume/Escritorio/Ultim_Semestre/Comp.ML/Practicas/project/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json"
tip_path="/home/jaume/Escritorio/Ultim_Semestre/Comp.ML/Practicas/project/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_tip.json"
user_path="/home/jaume/Escritorio/Ultim_Semestre/Comp.ML/Practicas/project/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json"


# %%
'''
with open(checkin_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

checkin = pd.DataFrame(data)'''
with open(business_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

business = pd.DataFrame(data)
with open(review_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

review = pd.DataFrame(data)
'''
with open(tip_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

tip = pd.DataFrame(data)
with open(user_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

user = pd.DataFrame(data)
'''

# %%
business_filtered = business[business["city"] == "Sparks"]
business_ids = business_filtered["business_id"]

filtered_reviews = review[review["business_id"].isin(business_ids)]
print(filtered_reviews.shape)

# %% [markdown]
# ### Remove users/business with low number of reviews

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of reviews per 'business_id' and 'user_id'
business_reviews = filtered_reviews['business_id'].value_counts().reset_index()
business_reviews.columns = ['business_id', 'review_count']

user_reviews = filtered_reviews['user_id'].value_counts().reset_index()
user_reviews.columns = ['user_id', 'review_count']

# Set up figure and axes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot for 'business_id'
sns.scatterplot(x=business_reviews.index, y=business_reviews['review_count'], ax=axes[0, 0])
axes[0, 0].set_title("Scatter Plot of Review Count per Business ID")
axes[0, 0].set_xlabel("Index")
axes[0, 0].set_ylabel("Number of Reviews")

# Histogram for 'business_id' review distribution
sns.histplot(business_reviews['review_count'], bins=1000, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Review Counts per Business ID")
axes[0, 1].set_xlabel("Number of Reviews")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_xlim(0, 100) 

# Scatter plot for 'user_id'
sns.scatterplot(x=user_reviews.index, y=user_reviews['review_count'], ax=axes[1, 0])
axes[1, 0].set_title("Scatter Plot of Review Count per User ID")
axes[1, 0].set_xlabel("Index")
axes[1, 0].set_ylabel("Number of Reviews")

# Histogram for 'user_id' review distribution
sns.histplot(user_reviews['review_count'], bins=1000, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Distribution of Review Counts per User ID")
axes[1, 1].set_xlabel("Number of Reviews")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_xlim(0, 10) 

# Adjust layout
plt.tight_layout()
plt.show()


# %%
# Define thresholds (adjust as needed)
business_threshold = 5  # Minimum number of reviews a business must have
user_threshold = 2  # Minimum number of reviews a user must have

# Count reviews per business and user
business_counts = filtered_reviews['business_id'].value_counts()
user_counts = filtered_reviews['user_id'].value_counts()

# Filter out businesses and users with low review counts
filtered_reviews = filtered_reviews[
    (filtered_reviews['business_id'].isin(business_counts[business_counts >= business_threshold].index)) &
    (filtered_reviews['user_id'].isin(user_counts[user_counts >= user_threshold].index))
]

# Display the filtered DataFrame
print(filtered_reviews.shape)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of reviews per 'business_id' and 'user_id'
business_reviews = filtered_reviews['business_id'].value_counts().reset_index()
business_reviews.columns = ['business_id', 'review_count']

user_reviews = filtered_reviews['user_id'].value_counts().reset_index()
user_reviews.columns = ['user_id', 'review_count']

# Set up figure and axes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot for 'business_id'
sns.scatterplot(x=business_reviews.index, y=business_reviews['review_count'], ax=axes[0, 0])
axes[0, 0].set_title("Scatter Plot of Review Count per Business ID")
axes[0, 0].set_xlabel("Index")
axes[0, 0].set_ylabel("Number of Reviews")

# Histogram for 'business_id' review distribution
sns.histplot(business_reviews['review_count'], bins=1000, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Review Counts per Business ID")
axes[0, 1].set_xlabel("Number of Reviews")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_xlim(0, 100) 

# Scatter plot for 'user_id'
sns.scatterplot(x=user_reviews.index, y=user_reviews['review_count'], ax=axes[1, 0])
axes[1, 0].set_title("Scatter Plot of Review Count per User ID")
axes[1, 0].set_xlabel("Index")
axes[1, 0].set_ylabel("Number of Reviews")

# Histogram for 'user_id' review distribution
sns.histplot(user_reviews['review_count'], bins=1000, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Distribution of Review Counts per User ID")
axes[1, 1].set_xlabel("Number of Reviews")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_xlim(0, 10) 

# Adjust layout
plt.tight_layout()
plt.show()


# %%
review_short= filtered_reviews.drop(columns=["review_id","business_id","date"])
review_short

# %% [markdown]
# ### Lowecasing

# %%
review_lower=review_short
review_lower["text"]=review_lower["text"].str.lower()
review_lower

# %%
corpus=review_lower.drop(columns=["useful","funny","cool"])

# %% [markdown]
# ### First Cleaning
# 

# %%
import string

corpus["text"] = corpus["text"].apply(lambda text: text.replace("\n", "").replace("...", ""))
corpus["text"] = corpus["text"].apply(lambda text: ''.join([char for char in text if char not in string.punctuation]))


# %% [markdown]
# ### Lemmatization

# %%
import spacy

nlp = spacy.load("en_core_web_sm")


# %%
corpus

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
# ### Remove stopWords

# %%
import nltk
import ast  # Para convertir strings en listas
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

corpus["words"] = corpus["words"].apply(lambda x: [word for word in x if word.lower() not in stop_words and not word.isdigit()])

# Limpiar directamente la columna 'words' eliminando signos de puntuación y saltos de línea

# %% [markdown]
# ### Remove foreing reviews

# %%
import pandas as pd
import nltk
from nltk.corpus import words

# Dowload english
nltk.download("words")
english_vocab = set(words.words())

# count words that are not in english
def count_non_english_words(word_list):
    return sum(1 for word in word_list if word.lower() not in english_vocab)

# Filtrar corpus
corpus["non_english_count"] = corpus["words"].apply(count_non_english_words)


pd.set_option('display.max_colwidth', None)  # Show full text without truncation
corpus


# %%
pd.reset_option('display.max_colwidth')

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Filter values up to 50
filtered_values = corpus["non_english_count"][corpus["non_english_count"] <= 100]

# Compute histogram and cumulative distribution
counts, bin_edges = np.histogram(filtered_values, bins=50, density=True)
cdf = np.cumsum(counts) / np.sum(counts)  # Normalize

# Find the threshold where 95% of the data accumulates
threshold_index = np.argmax(cdf >= 0.95)
threshold_value = bin_edges[threshold_index]

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Histogram (distribution)
sns.histplot(filtered_values, bins=30, kde=True, ax=axes[0], color="blue", alpha=0.6)
axes[0].set_xlabel("Non-English Count")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution of Non-English Count")

# Right plot: Cumulative Distribution Function (CDF)
axes[1].plot(bin_edges[1:], cdf, color="red", marker="o", linestyle="-", label="Cumulative Distribution")
axes[1].axvline(threshold_value, color="green", linestyle="--", label=f"95% accumulated ({threshold_value:.2f})")
axes[1].set_xlabel("Non-English Count")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Cumulative Distribution Function (CDF)")
axes[1].legend()

plt.tight_layout()
plt.show()

# Print the threshold value
print(f"The value where 95% of the data accumulates is approximately {threshold_value:.2f}")


# %%
corpus = corpus[
    (corpus["non_english_count"] <= threshold_value) &  # If the number or foreign words is higher than the threshold we set
    (corpus["non_english_count"] <= corpus["words"].apply(len) * 0.5)  # we remove the sample if a high percentage of the text is not in english
]


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
irrelevant_words=["well","tell","one","say","think","come","look","even","make","really","ask","use","see","want","go","order","good","work","place","need"]

# %%
corpus["words"] = corpus["words"].apply(lambda sublist: [word for word in sublist if word not in irrelevant_words])
corpus["words"] = corpus["words"].apply(lambda sublist: [word for word in sublist if word.strip()])


# %%
corpus['words_join'] = None
corpus.loc[:, "words_join"] = corpus["words"].apply(lambda x: " ".join([word for word in x if word]))

# %%
corpus=corpus.drop(columns="non_english_count")

# %%
corpus

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
# ## LDA

# %%
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
num_batches = 7
n_topics = 9
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
num_batches

# %%

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
# Suponiendo que tienes un DataFrame con una columna "words"
df_corpus = corpus


# Dividir en batches correctamente
num_batches = 7
df_batches = np.array_split(df_corpus, num_batches)  # Dividir el DataFrame en partes

# Rango de número de tópicos a probar
topic_range = range(2, 15)
coherence_values = np.zeros(len(topic_range))  # Acumular coherencia

# Procesar cada batch
for batch_idx, batch_df in enumerate(df_batches):
    print(f"\nProcesando Batch {batch_idx+1}/{num_batches}...")

    # Convertir la columna "words" a lista de listas
    batch_texts = corpus["words"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()

    # Crear diccionario y corpus para este batch
    dictionary = corpora.Dictionary(batch_texts)
    corpus2 = [dictionary.doc2bow(text) for text in batch_texts]

    # Calcular coherencia para cada número de tópicos
    for i, num_topics in enumerate(topic_range):
        lda_model = gensim.models.LdaModel(corpus=corpus2, id2word=dictionary, num_topics=num_topics, random_state=42)
        coherence_model = CoherenceModel(model=lda_model, texts=batch_texts, dictionary=dictionary, coherence='c_v')
        coherence_values[i] += coherence_model.get_coherence()  # Sumar coherencia del batch

# Promediar los valores de coherencia
coherence_values /= num_batches  

# Graficar coherencia vs. número de tópicos
plt.figure(figsize=(8, 5))
plt.plot(topic_range, coherence_values, marker='o')
plt.xlabel("Número de Tópicos")
plt.ylabel("Coherencia Promedio")
plt.title("Selección del Número de Tópicos con Coherencia (Batches)")
plt.grid()
plt.show()

# %%
corpus

# %%

n_topics = np.argmax(coherence_values)+2
num_batches=1
batches = np.array_split(corpus, num_batches)


# %%
n_topics=12

# %%
batches[0]

# %%
# Store topic distributions
all_topic_matrices = []

for batch_idx, batch in enumerate(batches):
    print(f"\nProcessing Batch {batch_idx+1}/{num_batches}...\n")
    
    # Feature extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(batch["words_join"].to_numpy())

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
repeated_words = {word for word, count in word_counts.items() if count > 2}

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


# %%
filtered_top_n_words_lda = [topic for topic in filtered_top_n_words_lda if topic]  


# %%
#filtered_top_n_words_lda=top_n_words_lda

# %%
filtered_top_n_words_lda = [
    [word for word in topic if word not in {"movie", "teather"}]
    for topic in filtered_top_n_words_lda
]


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
                 min_count=1,      # Ignora palabras con menos de 2 apariciones
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

    # Crear un color único para cada tópico
    topic_colors = {i: plt.cm.tab10(i) for i in range(len(topics))}

    word_to_topic = {}  # Diccionario para mapear palabra -> tópico
    for topic_idx, words in enumerate(topics):
        for word in words:
            word_to_topic[word] = topic_idx

    for i, label in enumerate(labels):
        x, y = vectors[i]
        topic_idx = word_to_topic.get(label, -1)
        color = topic_colors.get(topic_idx, "gray")
        plt.scatter(x, y, color=color, alpha=0.7)
        plt.text(x + 0.1, y + 0.1, label, fontsize=9, color=color)

    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.title("Visualización de palabras por tópico con t-SNE")
    plt.show()


# %%
print(list(model.wv.index_to_key)[:20])  # Muestra algunas palabras del vocabulario


# %%
# Extraer palabras únicas de filtered_top_n_words_lda
words_to_plot = list(set(word for topic in filtered_top_n_words_lda for word in topic))

# Reducir dimensionalidad
vectors_2d, labels = reduce_dimensions(model, num_dimensions=2, words=words_to_plot)

# Graficar con colores por tópico
plot_words_by_topic(vectors_2d, labels, filtered_top_n_words_lda)

# %%
from scipy.spatial.distance import euclidean
import numpy as np

# Obtener todos los vectores de palabras en los tópicos
word_vectors = {
    word: model.wv[word]
    for topic in filtered_top_n_words_lda
    for word in topic
    if word in model.wv
}

# Lista de palabras con vector disponible
word_list = list(word_vectors.keys())

# Calcular distancias entre todos los pares de palabras en los tópicos
distances = []
for topic in filtered_top_n_words_lda:
    topic_vectors = [word_vectors[word] for word in topic if word in word_vectors]
    for i in range(len(topic_vectors)):
        for j in range(i + 1, len(topic_vectors)):
            distances.append(euclidean(topic_vectors[i], topic_vectors[j]))

# Calcular la distancia media
mean_distance = np.mean(distances) if distances else 0
print(f"Distancia media entre palabras en los tópicos: {mean_distance}")



