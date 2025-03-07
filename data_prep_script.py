# %%
import pandas as pd
import json
import numpy as np


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

checkin = pd.DataFrame(data)
with open(business_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

business = pd.DataFrame(data)'''
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
review=review.head(1500)
review

# %%
review_short= review.drop(columns=["user_id","business_id","date"])
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


# %% [markdown]
# ### Remove Stopwords

# %%
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
corpus["words"] = corpus["words"].apply(lambda x: [word for word in x if word.lower() not in stop_words])



# %%
corpus

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus["text"]).toarray()

print(X.shape)

# %%
X

# %% [markdown]
# ### VADER POLARITY

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
corpus["text"].to_numpy()

# %%
corpus['words_join'] = None
corpus["words_join"] = corpus["words"].apply(lambda x: " ".join([word for word in x if word]))

# %% [markdown]
# ## LDA

# %%
from sklearn.feature_extraction.text import CountVectorizer
n_topics = 5
# feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus["words_join"].to_numpy())


# %%
from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(X)

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
lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)

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
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words

# %%
top_n_words_lda = get_top_n_words(10, lda_keys, X, vectorizer)

for i in range(len(top_n_words_lda)):
    print("Topic {}: ".format(i+1), top_n_words_lda[i])

# %%
corpus2=corpus
texts=corpus["words"].tolist()

# %%
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
# Crear diccionario y corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Rango de número de tópicos a probar
topic_range = range(2, 15)
coherence_values = []

# Entrenar modelos LDA y calcular coherencia
for num_topics in topic_range:
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherence_model.get_coherence())

# Graficar coherencia vs. número de tópicos
plt.figure(figsize=(8, 5))
plt.plot(topic_range, coherence_values, marker='o')
plt.xlabel("Número de Tópicos")
plt.ylabel("Coherencia")
plt.title("Selección del Número de Tópicos con Coherencia")
plt.grid()
plt.show()


