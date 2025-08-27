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

# %% Encontrado as palavras que não estão presentes no vacabulário e precisam ser substituidas


def get_words_to_replace(df):
    words = " ".join([word for line in df for word in line.lower().split()])

    doc = nlp(words)
    words_to_replace = set(
        [
            token.text
            for token in doc
            if token.is_oov
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.is_alpha
        ]
    )
    return words_to_replace


words_to_replace = list(get_words_to_replace(df["text_raw"]))

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

# %% LIMPEZA DAS POSTAGENS (PRÉ-PROCESSAMENTO)


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
def text_cleaner(tokens):
    text = [token.lemma_ for token in tokens if not token.is_stop]
    return " ".join(text)


df["text"] = df_tokens["tokens"].apply(text_cleaner)
df["text_list"] = df["text"].apply(
    lambda text: text.split() if isinstance(text, str) else []
)


# %% ANÁLISE DE PRIMEIRA E TERCEIRA PESSOA


def get_words_for_person(df, person: int = 1):
    words = [
        token.text
        for tokens in df
        for token in tokens
        if f"Person={person}" in token.morph
    ]
    return words


def person_viz(words):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=10
    ).generate(" ".join([word for word in words]))

    plt.figure(figsize=(15, 9), dpi=300, facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    count_words = Counter(words)

    plt.figure(figsize=(15, 9), dpi=300)
    plot = sns.barplot(
        x=[i[0] for i in count_words.most_common(20)],
        y=[i[1] for i in count_words.most_common(20)],
        palette="viridis",
        hue=[i[0] for i in count_words.most_common(20)],
        legend=False,
    )

    for container in plot.containers:
        plot.bar_label(container, padding=3, fontsize=10)

    plt.xlabel("Palavras", fontsize=11)
    plt.ylabel("Frequência", fontsize=11)
    plt.show()


words_1_person = get_words_for_person(df_tokens["tokens"], 1)
words_3_person = get_words_for_person(df_tokens["tokens"], 3)
person_viz(words_1_person)
person_viz(words_3_person)


def compare_persons(person_1, person_3, df):
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


compare_persons(words_1_person, words_3_person, df_tokens["tokens"])

# GERANDO AS VARIAVEIS DE 1 PESSOA E 3 PESSOA


def get_person_var(text):
    person_1, person_3 = 0, 0
    for token in text:
        if "Person=1" in token.morph:
            person_1 += 1
        elif "Person=3" in token.morph:
            person_3 += 1
    return [person_1, person_3]


df["person"] = df_tokens["tokens"].apply(get_person_var)


# %% ANÁLISE DE VERBOS


def get_verbs(df):
    words = [token.lemma_ for tokens in df for token in tokens if token.pos_ == "VERB"]
    return words


verbs = get_verbs(df_tokens["tokens"])

person_viz(verbs)

# %% PALAVRAS MAIS FREQUENTES (BAG OF WORDS)

bag_of_words = " ".join(df["text"])

count_of_words = Counter(bag_of_words.split())

words = [i[0] for i in count_of_words.most_common(20)]
frequency = [i[1] for i in count_of_words.most_common(20)]


# VISUALIZAÇÃO BAG OF WORDS
def words_viz(bag_of_words, words, frequency):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", min_font_size=10
    ).generate(bag_of_words)

    plt.figure(figsize=(15, 9), dpi=300, facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # GRÁFICO - 20 PALAVRAS MAIS FREQUENTES
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
    plt.ylabel("Frequencia", fontsize=11)
    plt.show()


words_viz(bag_of_words, words, frequency)

# %% DEFININDO N-GRAMS


def get_n_gram(n_gram):
    vectorizer = CountVectorizer(
        ngram_range=(n_gram, n_gram), stop_words=list(stopwords)
    )

    X = vectorizer.fit_transform(df["text"])

    feature_names = vectorizer.get_feature_names_out()

    df_n_gram_counts = pd.DataFrame(X.toarray(), columns=feature_names)

    frequencies = df_n_gram_counts.sum().sort_values(ascending=False)

    return frequencies


bigram_frequencies = get_n_gram(2)
trigram_frequencies = get_n_gram(3)


# %% VISUALIZAÇÃO N-GRAMS


def n_gram_viz(frequencies, n_gram):
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


n_gram_viz(bigram_frequencies, "Bigrama")
n_gram_viz(trigram_frequencies, "Trigrama")


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
def clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

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

    wcss_por_cluster = []

    for i in range(kmeans.n_clusters):
        pontos = X_array[labels == i]
        centroide = centroids[i]
        wcss_i = np.sum((pontos - centroide) ** 2)
        wcss_por_cluster.append(wcss_i)

    print(wcss_por_cluster)

    # feature_names = vectorizer.get_feature_names_out()

    # order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    # for i in range(n_clusters):
    #     print(f"\nCluster {i}:")
    #     print("Principais termos:", end="")
    #     for ind in order_centroids[i, :10]:
    #         print(f" {feature_names[ind]}", end="")
    #     print()


elbow_method(X_tfidf)
clustering(X_tfidf, 6)


# %% VETOR E KMEANS COMBINANDO TEXT COM PERSON

X_person = np.array(df["person"].tolist())

X_combined = hstack([X_tfidf, csr_matrix(X_person)])

elbow_method(X_combined)
clustering(X_combined, 4)
