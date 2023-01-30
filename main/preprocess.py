#!/usr/bin/env python3

from collections import Counter
import os
import epitran
import panphon
import re
import lzma
import tqdm
from emoji import is_emoji

# https://www.w3.org/TR/elreq/#ethiopic_punctuation
amharic_symb = '፠፡።፣፤፥፦፧፨‘’“”‹›«»€…'
# https://en.wikipedia.org/wiki/Bengali_alphabet#Punctuation_marks, https://en.wikipedia.org/wiki/Bengali_numerals
bengali_symb = '০১২৩৪৫৬৭৮৯৹৷৶৴৵৸₹–।'
english_punc = r'!"#$%&\'\(\)*\+,-./:;<=>?@\[\\\]^_`{|}~'
other_punc = r'‌'
punctuations = english_punc + amharic_symb + bengali_symb

ft = panphon.FeatureTable()


def save_lang(lang, tokens_all, vocab_ort, vocab_ipa):
    tokens_all = list(set(tokens_all))[:200_000]
    # sort by token form
    tokens_all.sort(key=lambda x: x[0])

    print(f'- number of tokens after cleaning up: {len(tokens_all)}')
    with open(f"data/ipa_tokens/{lang}.txt", 'w') as f:
        f.write('\n'.join([x[1] for x in tokens_all]))

    with open(f"data/multi.tsv", 'a') as f:
        f.write('\n'.join(["\t".join(x) for x in tokens_all]) + "\n")

    with open(f"data/vocab/ort_{lang}.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_ort))))

    with open(f"data/vocab/ipa_{lang}.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_ipa))))

    print(f'- saved {lang}')


def process_non_en(lang, ortho_name, min_freq=6):
    vocab_ipa = Counter()
    vocab_ort = Counter()
    print(f'\n=== Processing language: {lang} ===')
    token_counter = Counter()
    print(f'Gathering tokens for {lang}')

    with lzma.open(f'data/raw/{lang}.txt.xz') as f:
        for i, line in enumerate(tqdm.tqdm(f)):
            # take only first 10M lines
            if i > 10_000_000:
                break
            # take until we have 300k tokens (English has 125k) - they will be cut down to 200k later
            if i % 100_000 == 0 and len([k for k, v in token_counter.items() if v >= min_freq]) >= 300_000:
                break
            tokens = line.decode('utf-8').split()
            # strip punctuation around tokens
            tokens = [token.strip(punctuations).strip() for token in tokens]
            token_counter.update(tokens)

    print(f"- loaded {i} lines")
    print('- number of tokens', len(token_counter))
    frequent_tokens = [k for k, v in token_counter.items() if v >= min_freq]

    print(
        f'- number of tokens with min {min_freq} occurrences: {len(frequent_tokens)}',
    )

    epi = epitran.Epitran(ortho_name)

    tokens_all = []
    print('- converting to IPA')
    for token in tqdm.tqdm(frequent_tokens):
        if any(c in "0123456789" or is_emoji(c) for c in token):
            continue

        # doing ''.join(ipa_segs(s)) removes non-ipa characters from the string
        segments = ft.ipa_segs(epi.transliterate(token))
        if segments:
            vocab_ort.update(token)
            vocab_ipa.update(segments)

            tokens_all.append((
                token,
                ''.join(segments),
                lang,
                # CMU pronunciation is not possible
                ""
            ))

    # accept all IPA
    vocab_ipa = {k for k, v in vocab_ipa.most_common()}
    # accept only frequent enough characters into the vocabulary
    print("- size of vocab ort before pruning", len(vocab_ort))
    vocab_ort = {
        k
        for k, v in vocab_ort.most_common()
        if v >= 20 and not k.isspace() and len(k) > 0
    } | {"😕"}
    print("- size vocab ort after pruning", len(vocab_ort))

    # filter bad characters and use unsure emoji in their place
    tokens_all = [
        (
            "".join([c if c in vocab_ort else "😕" for c in token]),
            ipa,
            lang,
            pronunciation,
        )
        for token, ipa, lang, pronunciation in tokens_all
    ]

    save_lang(lang, tokens_all, vocab_ort, vocab_ipa)

    return vocab_ort, vocab_ipa


def process_en():
    cmu_pronunciation = [
        x.rstrip("\n").split("  ")
        for x in open('data/raw/cmudict-0.7b.txt')
        if x[0] != ";"
    ]
    cmu_pronunciation = {x[0]: x[1] for x in cmu_pronunciation}

    tokens_all = []
    vocab_ipa = set()
    vocab_ort = set()
    print('Gathering tokens for en')
    with open('data/raw/cmudict-0.7b-ipa.txt') as f:
        for line in tqdm.tqdm(f):
            if line[0].isalpha():
                token, ipa = line.rstrip("\n").split('\t')

                # if multiple pronunciations, take first one
                ipa = ipa.split(' ')[0]
                # remove stress marks and commas
                ipa = re.sub('[,ˌˈ]', '', ipa)
                segments = ft.ipa_segs(ipa)

                vocab_ipa.update(segments)
                if segments and token in cmu_pronunciation:
                    pronunciation = cmu_pronunciation[token]
                    token = token.lower()
                    vocab_ort.update(token)

                    # append a triplet of (token, ipa, cmu pronunciation)
                    tokens_all.append((
                        token,
                        ''.join(segments),
                        "en",
                        pronunciation,
                    ))

    save_lang("en", tokens_all, vocab_ort, vocab_ipa)
    return vocab_ort, vocab_ipa


if __name__ == '__main__':
    # clear multi file
    open(f"data/multi.tsv", 'w').close()
    os.makedirs("data/vocab/", exist_ok=True)
    os.makedirs("data/ipa_tokens/", exist_ok=True)

    vocab_ort_all = set()
    vocab_ipa_all = set()

    # for english, we use the IPA version of the CMU pronunciation dict from
    # https://github.com/menelik3/cmudict-ipa
    vocab_ort, vocab_ipa = process_en()
    vocab_ort_all |= vocab_ort
    vocab_ipa_all |= vocab_ipa

    LANG_TO_ORTHO = {
        'am': 'amh-Ethi',
        'bn': 'ben-Beng',
        'uz': 'uzb-Latn',
        'pl': 'pol-Latn',
        'es': 'spa-Latn',
        'sw': 'swa-Latn',
    }
    for lang, ortho_name in LANG_TO_ORTHO.items():
        vocab_ort, vocab_ipa = process_non_en(lang, ortho_name)
        vocab_ort_all |= vocab_ort
        vocab_ipa_all |= vocab_ipa

    with open(f"data/vocab/ipa_multi.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_ipa_all))))
    with open(f"data/vocab/ort_multi.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_ort_all))))

    print()
    print("Multi vocab file generated")
