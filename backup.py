# %% PALAVRAS MAIS IMPORTANTES POR CLUSTER:

feature_names = vectorizer.get_feature_names_out()

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print("Principais termos:", end="")
    for ind in order_centroids[i, :10]:
        print(f" {feature_names[ind]}", end="")
    print()  # -*- coding: utf-8 -*-

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
