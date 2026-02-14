import pandas as pd

df = pd.read_csv("breast-cancer.csv")

print("Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nValue counts of each column:\n")

for col in df.columns:
    print(col)
    print(df[col].value_counts().head())
    print("-" * 40)