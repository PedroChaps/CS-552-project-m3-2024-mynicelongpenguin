import random

random.seed(42)

data = []
with open("test_dataset.jsonl", "r") as f:
    for line in f:
        data.append(line)

random.shuffle(data)

data = data[:5000]

with open("test_dataset_min.jsonl", "w") as f:
    f.write("".join(data))

