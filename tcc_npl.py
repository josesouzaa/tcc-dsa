# %%

#############################################################################
# 4 - NLP
#############################################################################

# !pip install pandas
# !pip install seaborn
# !pip install matplotlib
# !pip install wordcloud
# !pip install spacy
# !pip install sklearn
# !python -m spacy download pt_core_news_sm

# %% IMPORTANDO OS PACOTES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

import spacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import hstack, csr_matrix

# %% INSTANCIANDO O SPACY E IMPORTANDO O BANCO DE DADOS

nlp = spacy.load("pt_core_news_md")

df = pd.read_csv("db.csv")

# %% AJUSTANDO O BANCO DE DADOS

df.drop(
    columns=[
        "user_name",
        "user_at",
        "user_url",
        "user_avatar_url",
        "images_url",
        "post_url",
    ],
    inplace=True,
)

df["text_raw"] = df["text_raw"].fillna("").astype(str)


# %% DEFININDO E ADICIONANDO STOPWORDS E OBJETO DE SUBSTITUIÇÕES

stopwords = nlp.Defaults.stop_words
new_stopwords = {"pra", "to", "ja", "ta", "so", "ai", "pro", "tao", "sera", "la"}
for word in new_stopwords:
    stopwords.add(word)

substitutions = {
    "vc": "você",
    "pq": "porque",
    "pque": "porque",
    "tbm": "também",
    "tb": "também",
    "blz": "beleza",
    "vcs": "vocês",
    "tô": "estou",
    "tá": "esta",
    "dnv": "de novo",
    "hj": "hoje",
    "pf": "por favor",
    "pfv": "por favor",
    "pra": "para",
}

# %% PRÉ-PROCESSAMENTO


# Tokenizando as palavras e fazendo substituições
def get_tokens(text):
    if not isinstance(text, str):
        return []
    new_text = " ".join(
        [substitutions.get(word, word) for word in text.lower().split()]
    )
    tokens = [
        token
        for token in nlp(new_text)
        if not token.is_punct and not token.is_space and token.is_alpha
    ]
    return tokens


df_tokens = pd.DataFrame()
df_tokens["tokens"] = df["text_raw"].apply(get_tokens)


# Retirando Stopwrds e gerando o texto lemmatizado
def preprocess_text(tokens):
    text = [token.lemma_ for token in tokens if not token.is_stop]
    return " ".join(text)


df["text"] = df_tokens["tokens"].apply(preprocess_text)
df["text_list"] = df["text"].apply(
    lambda text: text.split() if isinstance(text, str) else []
)

# %% PALAVRAS MAIS FREQUENTES (BAG OF WORDS)


def prepare_words_data(text_series):
    bag_of_words = " ".join(text_series)

    count_of_words = Counter(bag_of_words.split())

    words = [i[0] for i in count_of_words.most_common(20)]
    frequency = [i[1] for i in count_of_words.most_common(20)]
    return bag_of_words, words, frequency


def visualize_words_analysis(text_series):
    bag_of_words, words, frequency = prepare_words_data(text_series)

    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=10
    ).generate(bag_of_words)

    plt.figure(figsize=(15, 9), dpi=300, facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.figure(figsize=(15, 9), dpi=300)
    plot = sns.barplot(
        x=words,
        y=frequency,
        palette="viridis",
        hue=words,
        legend=False,
    )

    for container in plot.containers:
        plot.bar_label(container, padding=3, fontsize=10)

    plt.xlabel("Palavras", fontsize=11)
    plt.ylabel("Frequência", fontsize=11)
    plt.show()


visualize_words_analysis(df["text"])


# %% ANÁLISE DE PRIMEIRA E TERCEIRA PESSOA


def get_words_for_person(df, person: int = 1):
    words = [
        token.text
        for tokens in df
        for token in tokens
        if f"Person={person}" in token.morph
    ]
    return words


words_1_person = get_words_for_person(df_tokens["tokens"], 1)
words_3_person = get_words_for_person(df_tokens["tokens"], 3)
visualize_words_analysis(words_1_person)
visualize_words_analysis(words_3_person)


def visualize_person_distribution(person_1, person_3, df):
    words = [
        token.text
        for tokens in df
        for token in tokens
        if not "Person=1" in token.morph and not "Person=3" in token.morph
    ]

    plt.pie(
        [len(person_1), len(person_3), len(words)],
        labels=[
            f"1 Pessoa, {len(person_1)}",
            f"3 Pessoa, {len(person_3)}",
            f"Demais palavras, {len(words)}",
        ],
        autopct="%.0f%%",
    )
    plt.show()


visualize_person_distribution(words_1_person, words_3_person, df_tokens["tokens"])

# GERANDO A VARIÁVEL DE 1 PESSOA E 3 PESSOA


def count_person_usage(tokens):
    person_1, person_3 = 0, 0
    for token in tokens:
        if "Person=1" in token.morph:
            person_1 += 1
        elif "Person=3" in token.morph:
            person_3 += 1
    return [person_1, person_3]


df["person"] = df_tokens["tokens"].apply(count_person_usage)


# %% ANÁLISE DE VERBOS


def get_verbs(df):
    verbs = [token.lemma_ for tokens in df for token in tokens if token.pos_ == "VERB"]
    return verbs


verbs = get_verbs(df_tokens["tokens"])

visualize_words_analysis(verbs)


# %% DEFININDO N-GRAMS


def get_ngram_frequencies(ngram):
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram), stop_words=list(stopwords))

    X = vectorizer.fit_transform(df["text"])

    feature_names = vectorizer.get_feature_names_out()

    df_ngram_counts = pd.DataFrame(X.toarray(), columns=feature_names)

    frequencies = df_ngram_counts.sum().sort_values(ascending=False)

    return frequencies


bigram_frequencies = get_ngram_frequencies(2)
trigram_frequencies = get_ngram_frequencies(3)


# %% VISUALIZAÇÃO N-GRAMS


def visualize_ngram_analysis(frequencies, n_gram):
    plt.figure(figsize=(15, 9), dpi=300)
    plot = sns.barplot(
        x=frequencies.head(10).values,
        y=frequencies.head(10).index,
        palette="viridis",
        hue=frequencies.head(10).values,
        legend=False,
    )

    for container in plot.containers:
        plot.bar_label(container, padding=3, fontsize=10)

    plt.xlabel("Frequência", fontsize=11)
    plt.ylabel(f"{n_gram}", fontsize=11)
    plt.show()


visualize_ngram_analysis(bigram_frequencies, "Bigrama")
visualize_ngram_analysis(trigram_frequencies, "Trigrama")


# %% VETOR E KMEANS SOMENTE TEXTO

vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8)

X_tfidf = vectorizer.fit_transform(df["text"])


# MÉTODO DO COTOVELO PARA SABER QUANTOS CLUSTERS USAR
def elbow_method(X):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("WCSS")
    plt.xticks(range(1, 11))
    plt.show()


# CLUSTERIZANDO AS POSTAGENS
def perform_cluster_analysis(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    results = {}
    cluster_words_data = {}

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        _, words, frequency = prepare_words_data(cluster_df["text"])

        cluster_words_data[cluster_id] = {
            "num_items": cluster_df.shape[0],
            "words": words,
            "frequency": frequency,
        }

    results["cluster_words_data"] = cluster_words_data

    # MÉDIA DE PALAVRAS EM 1 E 3 PESSOA POR CLUSTER
    df["person_1"] = df["person"].apply(lambda x: x[0])
    df["person_3"] = df["person"].apply(lambda x: x[1])

    person_1_avg = df.groupby("cluster")["person_1"].mean().to_dict()
    person_3_avg = df.groupby("cluster")["person_3"].mean().to_dict()

    results["person_1_avg"] = person_1_avg
    results["person_3_avg"] = person_3_avg

    # GRÁFICO - NÚMERO DE POSTAGENS POR CLUSTER
    plt.figure(figsize=(15, 9), dpi=300)
    plot = sns.barplot(
        data=df.groupby("cluster").count(),
        x="cluster",
        y="text",
        palette="viridis",
        hue="cluster",
        legend=False,
    )

    for container in plot.containers:
        plot.bar_label(container, padding=3, fontsize=10)

    plt.xlabel("Cluster", fontsize=15)
    plt.ylabel("Total de postagens", fontsize=15)
    plt.show()

    # WSS POR CLUSTER
    X_array = X.toarray()
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    wcss_for_cluster = []

    for i in range(kmeans.n_clusters):
        pontos = X_array[labels == i]
        centroide = centroids[i]
        wcss_i = np.sum((pontos - centroide) ** 2)
        wcss_for_cluster.append(wcss_i)

    results["wcss_for_cluster"] = wcss_for_cluster

    return results


elbow_method(X_tfidf)
text_clustering_results = perform_cluster_analysis(X_tfidf, 6)


# %% VETOR E KMEANS COMBINANDO TEXT COM PERSON

X_person = np.array(df["person"].tolist())

X_combined = hstack([X_tfidf, csr_matrix(X_person)])

elbow_method(X_combined)
text_and_person_clustering_results = perform_cluster_analysis(X_combined, 4)
