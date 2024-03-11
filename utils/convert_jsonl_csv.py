import pandas as pd
import json

jsonl_file_path = "out/time_bench_result.jsonl"
csv_file_path = "out/time_bench_result.csv"

data = []
with open(jsonl_file_path, "r") as jsonl_file:
    for line in jsonl_file:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.to_csv(csv_file_path, index=False)
