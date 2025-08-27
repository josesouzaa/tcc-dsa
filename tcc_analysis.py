# %%

# !pip install pandas
# !pip install seaborn
# !pip install matplotlib

# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# %%

df = pd.read_csv("db.csv")

# %%

plt.figure(figsize=(15, 9), dpi=300)
plot1 = sns.barplot(
    data=df[["year_month", "interactions"]].groupby("year_month").sum(),
    x="year_month",
    y="interactions",
    palette="viridis",
    hue="year_month",
    legend=False,
)

for container in plot1.containers:
    plot1.bar_label(container, padding=3, fontsize=10)

plt.xlabel("Meses", fontsize=15)
plt.ylabel("Interações totais", fontsize=15)
plt.show()

# %%

plt.figure(figsize=(15, 9), dpi=300)
plot2 = sns.barplot(
    data=df.groupby("year_month").count(),
    x="year_month",
    y="user_at",
    palette="viridis",
    hue="year_month",
    legend=False,
)

for container in plot2.containers:
    plot2.bar_label(container, padding=3, fontsize=10)

plt.xlabel("Meses", fontsize=15)
plt.ylabel("Total de posts", fontsize=15)
plt.show()

# %%

plt.figure(figsize=(15, 9), dpi=300)
plot3 = sns.barplot(
    data=df[["period", "interactions"]].groupby("period").sum(),
    x="period",
    y="interactions",
    palette="viridis",
    hue="period",
    legend=False,
)
for container in plot3.containers:
    plot3.bar_label(container, fmt="%.0f", padding=3, fontsize=10)


def format_y_axis(value, TickerNumber):
    return "{:.0f}".format(value)


plot3.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.xlabel("Periodos", fontsize=15)
plt.ylabel("Interações totais", fontsize=15)
plt.show()

# %%

plt.figure(figsize=(15, 9), dpi=300)
plot4 = sns.barplot(
    data=df.groupby("period").count(),
    x="period",
    y="user_at",
    palette="viridis",
    hue="period",
    legend=False,
)

for container in plot4.containers:
    plot4.bar_label(container, padding=3, fontsize=10)

plt.xlabel("Periodos", fontsize=15)
plt.ylabel("Total de posts", fontsize=15)
plt.show()

# %%

plt.figure(figsize=(15, 9), dpi=300)
plot5 = sns.barplot(
    data=df.groupby("has_image").count(),
    x="has_image",
    y="user_at",
    palette="viridis",
    hue="period",
    legend=False,
)

for container in plot5.containers:
    plot5.bar_label(container, padding=3, fontsize=10)

plot5.set_xticks([0, 1])
plot5.set_xticklabels(["Não", "Sim"], fontsize=15)
plt.xlabel("Possui imagem", fontsize=15)
plt.ylabel("Total de posts", fontsize=15)
plt.show()
