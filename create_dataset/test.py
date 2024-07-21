import json
import collections
import numpy as np

data = [json.loads(x) for x in open("data/train.jsonl", "r")]

data_clusters = collections.defaultdict(list)
for token in data:
    data_clusters[token["token_ort"]].append(token)

disprepancies = []
for cluster in data_clusters.values():
    if len(cluster) > 1:
        if any([
            token["purpose"] != cluster[0]["purpose"] and (
                (token["token_ipa"] == cluster[0]["token_ipa"]) !=
                (token["token_arp"] == cluster[0]["token_arp"])
            )
            for token in cluster[1:]
        ]):
            disprepancies.append(True)
            print(cluster)
            print()
        else:
            disprepancies.append(False)

print(f"Cluster disprepancies")
print(f"- Cluster count:     {len(disprepancies)}")
print(f"- Disprepancy count: {sum(disprepancies)} ({np.average(disprepancies):.2%})")


# Cluster disprepancies
# - Cluster count:     72677
# - Disprepancy count: 331 (0.46%)