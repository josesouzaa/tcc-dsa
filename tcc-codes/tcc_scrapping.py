# %% INSTALANDO OS PACOTES

# !pip install pandas
# !pip install selenium

# %% IMPORTANDO OS PACOTES

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# %% CRIANDO O DATAFRAME

colunas = [
    "user_name",
    "user_at",
    "user_url",
    "user_avatar_url",
    "date",
    "text",
    "images_url",
    "replys",
    "reposts",
    "quotes",
    "likes",
    "post_url",
]

df = pd.DataFrame(columns=colunas)


# %% DEFININDO URL + CONTROLANDO O NAVEGADOR

url = "https://nitter.net/DuolingoBrasil"

navegador = webdriver.Chrome()

navegador.get(url)

navegador.maximize_window()

# %% FUNÇÃO DE PASSAR A PÁGINA


def get_next_page():
    btn = navegador.find_element(By.CSS_SELECTOR, "div.show-more a[href^='?cursor=']")

    navegador.execute_script("arguments[0].scrollIntoView()", btn)

    WebDriverWait(navegador, 10).until(EC.element_to_be_clickable(btn))

    btn.click()


# %% FUNÇÃO DE PERCORRER OS POSTS E ADICIONÁ-LOS AO BANCO DE DADOS


def get_posts(df):
    posts = navegador.find_elements(By.XPATH, "//div[@class='timeline-item ']")

    for post in posts:
        # Elementos unitários
        user_name = post.find_element(By.CSS_SELECTOR, "a.fullname").text
        user_at = post.find_element(By.CSS_SELECTOR, "a.username").text
        user_url = post.find_element(By.CSS_SELECTOR, "a.username").get_attribute(
            "href"
        )
        user_avatar_url = post.find_element(
            By.CSS_SELECTOR, "a.tweet-avatar img"
        ).get_attribute("src")
        date = post.find_element(By.CSS_SELECTOR, "span.tweet-date a").get_attribute(
            "title"
        )
        text = post.find_element(By.CSS_SELECTOR, "div.tweet-content").text
        post_url = post.find_element(By.CSS_SELECTOR, "a.tweet-link").get_attribute(
            "href"
        )

        # Elementos em lista
        stats = [
            stat.text
            for stat in post.find_elements(By.CSS_SELECTOR, "span.tweet-stat div")
        ]
        images = [
            img.get_attribute("href")
            for img in post.find_elements(
                By.XPATH,
                ".//div[contains(@class, 'attachment') and contains(@class, 'image') and not(ancestor::*[contains(@class, 'quote')])]/a[contains(@class, 'still-image')]",
            )
        ]

        if user_at == "@DuolingoBrasil":
            df_post = pd.DataFrame(
                [
                    [
                        user_name,
                        user_at,
                        user_url,
                        user_avatar_url,
                        date,
                        text,
                        images,
                        stats[0],
                        stats[1],
                        stats[2],
                        stats[3],
                        post_url,
                    ]
                ],
                columns=colunas,
            )
            df = pd.concat([df, df_post], ignore_index=True)

    return df


# %% EXECUTANDO AS FUNÇÕES E WHILE

df = get_posts(df)
get_next_page()

while "2023" not in df.tail(1)["date"].values[0]:
    df = get_posts(df)
    get_next_page()

# %% SALVANDO O BANCO DE DADOS

df.to_csv("db_raw.csv", index=False, encoding="utf-8")
# df.to_excel("db_raw.xlsx", index=False)
