# %% INSTALANDO OS PACOTES

# !pip install pandas
# !pip install ast

# %% IMPORTANDO OS PACOTES

import pandas as pd
import ast

# %% IMPORTANDO O BANCO DE DADOS E RENOMEANDO ALGUMAS COLUNAS

df = pd.read_csv("db_raw.csv")
df = df.rename(columns={"date": "date_raw", "text": "text_raw"})

# %% TRANSFORMANDO AS COLUNAS DE INTERAÇÕES EM NÚMEROS INTEIROS E CRIANDO A COLUNA "INTERACTIONS" SOMANDO TODAS AS INTERAÇÕES


def cols_to_number(cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "")).fillna(0).astype("int64")

    df["interactions"] = (
        df[["replys", "reposts", "quotes", "likes"]].sum(axis=1).astype("int64")
    )


cols_to_number(["replys", "reposts", "quotes", "likes"])

# %% VERIFICANDO SE O POST POSSUÍ IMAGEM OU NÃO E CRIANDO UMA COLUNA COM ESSA INFORMAÇÃO

df["images_url"] = df["images_url"].apply(ast.literal_eval)
df["has_image"] = df["images_url"].apply(lambda x: 1 if len(x) > 0 else 0)
df["has_image"] = df["has_image"].astype("category")

# %% FORMATANDO A DATA E ATRIBUINDO PERIODO PARA CADA OBSERVAÇÃO


def get_period(hour):
    if 0 <= hour < 6:
        return "madrugada"
    elif 6 <= hour < 12:
        return "manhã"
    elif 12 <= hour < 18:
        return "tarde"
    else:
        return "noite"


df["date_raw"] = pd.to_datetime(df["date_raw"], format="%b %d, %Y · %I:%M %p %Z")
df["date"] = df["date_raw"].dt.strftime("%d/%m/%Y")
df["year_month"] = df["date_raw"].dt.strftime("%Y/%m")
df["period"] = df["date_raw"].dt.hour.apply(get_period)
df.drop(columns=["date_raw"], inplace=True)

# %% SALVANDO O BANCO DE DADOS

df.to_csv("db.csv", index=False, encoding="utf-8")
