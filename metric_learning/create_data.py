#!/usr/bin/env python3

import panphon2

f = panphon2.FeatureTable()

epi = epitran.Epitran(code="ces-Latn")
w = epi.transliterate("ahoy")

f.word_to_vectors(w)