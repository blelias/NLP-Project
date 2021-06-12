# %%
import pandas as pd
import matplotlib as plt
import seaborn as sns
# %%
df = pd.read_csv("cleaned_dataset.csv")

# %%
df["verse_cleaned"] = df["verse_cleaned"].dropna()
# %%
df["verse_cleaned"].isna().sum()
# %%
# EDA
sns.boxplot(len(df["verse_cleaned"].str.len()))
# %%
list(df["verse_cleaned"].str.len())
df.columns
# %%
df["verse_cleaned"]