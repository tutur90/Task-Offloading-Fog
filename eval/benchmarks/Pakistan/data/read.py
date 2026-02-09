import pandas as pd
import json


df = json.load(open("/Users/arthurgaron/Code/Task-Offloading-Fog/eval/benchmarks/Pakistan/data/Tuple30K/Tuple30K.json", "r", encoding="utf-8-sig"))

df = pd.DataFrame(df)

df["CreationTime"] = pd.to_datetime(df["CreationTime"])

print(df.columns)

df = df[['Size', 'Name', 'MIPS', 'RAM', 'BW', "CreationTime", 'DataType', 'DeviceType']].sort_values("CreationTime").reset_index(drop=True)

print(df.head(20))


# print(df.groupby("CreationTime").size())
print(df[['Size', 'MIPS', 'RAM', 'BW', "CreationTime"]].groupby("CreationTime").sum())
# print(df.describe())