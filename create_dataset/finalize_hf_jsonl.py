import json

data = [line.rstrip().split("\t") for line in open("data/multi_3.tsv", "r")]

with open("data/train.jsonl", "w") as f:
    f.write("\n".join([json.dumps({
        "token_ort": x[0],
        "token_ipa": x[1],
        "token_arp": x[4],
        "lang": x[2],
        "purpose": x[3],
    }, ensure_ascii=False) for x in data]))