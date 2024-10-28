import json

data = [line.rstrip().split("\t") for line in open("data/multi_3.tsv", "r")]

with open("data/train.jsonl", "w") as f:
    for x in data:
        if x[3].startswith("analogy_"):
            extra_index = "_".join(x[3].split("_")[1:])
            x[3] = "analogy"
        elif x[3].startswith("cognate_"):
            extra_index = "_".join(x[3].split("_")[1:])
            x[3] = "cognate"
        else:
            extra_index = ""

        f.write(json.dumps({
            "token_ort": x[0],
            "token_ipa": x[1],
            "token_arp": x[4],
            "lang": x[2],
            "purpose": x[3],
            "extra_index": extra_index,
        }, ensure_ascii=False)+"\n")