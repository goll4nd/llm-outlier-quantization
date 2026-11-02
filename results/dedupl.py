import pandas as pd

df = pd.read_csv("results/baseline.csv")
before = len(df)
df = df.drop_duplicates(subset=["model", "bits", "group_size", "sym"], keep="last")
df.to_csv("results/baseline.csv", index=False)
print(f"{before} -> {len(df)} строк")