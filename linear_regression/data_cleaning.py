import pandas as pd

df = pd.read_csv("Mumbai1.csv")


# Drop the column using the index (e.g., index 0)
df.drop(df.columns[0], axis=1, inplace=True)

# Apply One-Hot Encoding using pd.get_dummies() on the 'Place' column
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)

df_encoded = df_encoded.astype(int)

print(df_encoded)


