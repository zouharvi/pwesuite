#!/usr/bin/env python3

from collections import Counter
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


def save_lang(lang, vocab_all, vocab_chars):
    vocab_all = list(set(vocab_all))
    # sort by token form
    vocab_all.sort(key=lambda x: x[0])

    print(f'- number of tokens after cleaning up: {len(vocab_all)}')
    with open(f"data/ipa_tokens_{lang}.txt", 'w') as f:
        f.write('\n'.join([x[1] for x in vocab_all]))

    with open(f"data/multi.tsv", 'a') as f:
        f.write('\n'.join(["\t".join(x) for x in vocab_all])+"\n")

    with open(f"data/vocab_{lang}.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_chars))))

    print(f'- saved {lang}')


def process_non_en(lang, ortho_name, min_freq=5):
    vocab_chars = set()
    print(f'\n=== Processing language: {lang} ===')
    vocab_counter = Counter()
    print(f'Gathering tokens for {lang}')

    with lzma.open(f'data/raw/{lang}.txt.xz') as f:
        for i, line in enumerate(tqdm.tqdm(f)):
            # take only first 10M lines
            if i > 10_000_000:
                break
            # take until we have 250k tokens (English has 125k) - they will be cut down to 200k later
            if i % 100_000 == 0 and len([k for k, v in vocab_counter.items() if v >= min_freq]) >= 250_000:
                break
            tokens = line.decode('utf-8').split()
            # strip punctuation around tokens
            tokens = [token.strip(punctuations) for token in tokens]
            vocab_counter.update(tokens)

    print(f"- loaded {i} lines")
    print('- number of tokens', len(vocab_counter))
    frequent_tokens = [k for k, v in vocab_counter.items() if v >= min_freq]

    print(
        f'- number of tokens with min {min_freq} occurrences: {len(frequent_tokens)}',
    )

    epi = epitran.Epitran(ortho_name)

    vocab_all = []
    print('- converting to IPA')
    for token in tqdm.tqdm(frequent_tokens):
        if any(c in "0123456789" or is_emoji(c) for c in token):
            continue
        # doing ''.join(ipa_segs(s)) removes non-ipa characters from the string
        segments = ft.ipa_segs(epi.transliterate(token))
        vocab_chars.update(segments)
        if segments:
            vocab_all.append((
                token,
                ''.join(segments),
                lang,
                # CMU pronunciation is not possible
                ""
            ))
            if len(vocab_all) >= 200_000:
                break

    save_lang(lang, vocab_all, vocab_chars)


def process_en():
    cmu_pronunciation = [
        x.rstrip("\n").split("  ")
        for x in open('data/raw/cmudict-0.7b.txt')
        if x[0] != ";"
    ]
    cmu_pronunciation = {x[0]: x[1] for x in cmu_pronunciation}

    vocab_all = []
    vocab_chars = set()
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

                vocab_chars.update(segments)
                if segments and token in cmu_pronunciation:
                    # append a triplet of (token, ipa, cmu pronunciation)
                    vocab_all.append((
                        token,
                        ''.join(segments),
                        "en",
                        cmu_pronunciation[token],
                    ))

    save_lang("en", vocab_all, vocab_chars)


if __name__ == '__main__':
    # clear multi file
    open(f"data/multi.tsv", 'w').close()

    # for english, we use the IPA version of the CMU pronunciation dict from
    # https://github.com/menelik3/cmudict-ipa
    process_en()

    LANG_TO_ORTHO = {
        'am': 'amh-Ethi',
        'bn': 'ben-Beng',
        'uz': 'uzb-Latn',
        'pl': 'pol-Latn',
        'es': 'spa-Latn',
        'sw': 'swa-Latn',
    }
    for lang, ortho_name in LANG_TO_ORTHO.items():
        process_non_en(lang, ortho_name)

    # consolidate vocab
    vocab_multi = set()
    langs = ['en'] + list(LANG_TO_ORTHO.keys())
    for lang in langs:
        with open(f"data/vocab_{lang}.txt") as f:
            vocab_multi.update(f.read().split())

    print()

    with open(f"data/vocab_multi.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_multi))))

    print("Multi vocab file generated")
