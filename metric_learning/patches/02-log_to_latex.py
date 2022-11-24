#!/usr/bin/env python3

TXT = """Train loss  0.43408 Dev loss  0.29050
Evaluating correlation for dev
Evaluating correlation for train
Train pearson  94.31% Train spearman 94.25%
Dev pearson    94.52% Dev spearman   94.10%


""".strip()

for line in TXT.split("\n"):
    if "Dev loss" in line:
        line = line.split("Dev loss")
        train_loss = float(line[0].strip().split(" ")[-1])
        dev_loss = float(line[1].strip().split(" ")[-1])
    if "Train spearman" in line:
        line = line.split("Train spearman")
        train_corr_pearson = float(
            line[0].strip().split(" ")[-1].rstrip("%")
        ) / 100
        train_corr_spearman = float(
            line[1].strip().split(" ")[-1].rstrip("%")
        ) / 100
    if "Dev spearman" in line:
        line = line.split("Dev spearman")
        dev_corr_pearson = float(
            line[0].strip().split(" ")[-1].rstrip("%")
        )/100
        dev_corr_spearman = float(
            line[1].strip().split(" ")[-1].rstrip("%")
        )/100


print(
    f"{train_corr_pearson*100:.1f}\\%",
    f"{train_corr_spearman*100:.1f}\\%",
    f"{train_loss:.3f}",
    f"{dev_corr_pearson*100:.1f}\\%",
    f"{dev_corr_spearman*100:.1f}\\%",
    f"{dev_loss:.3f}",
    sep=" & "
)
