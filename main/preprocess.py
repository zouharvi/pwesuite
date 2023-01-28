#!/usr/bin/env python3

from collections import Counter
import epitran
import panphon
import re
from tqdm import tqdm
import lzma

# https://www.w3.org/TR/elreq/#ethiopic_punctuation
amharic_symb = '፠፡።፣፤፥፦፧፨‘’“”‹›«»€…'
# https://en.wikipedia.org/wiki/Bengali_alphabet#Punctuation_marks, https://en.wikipedia.org/wiki/Bengali_numerals
bengali_symb = '০১২৩৪৫৬৭৮৯৹৷৶৴৵৸₹–।'
english_punc = r'!"#$%&\'\(\)*\+,-./:;<=>?@\[\\\]^_`{|}~'
other_punc = r'‌'
punctuations = english_punc + amharic_symb + bengali_symb

ft = panphon.FeatureTable()

def transliterate_lang(lang, ortho_name, min_occur=5):
    vocab = set()
    print(f'=== Processing language: {lang} ===')
    vocab_counter = Counter()
    fpath = f'data/raw/{lang}.txt.xz'
    print(f'Gathering tokens for {lang}...')
    # num_lines = sum(1 for line in lzma.open(fpath))
    with lzma.open(fpath) as f:
        for i, line in enumerate(tqdm(f)):
            # tokens = re.sub(f'[0-9{punctuations}]', ' ', line.decode('utf-8')).split()
            if i > 1e7:
                break
            tokens = line.decode('utf-8').split()
            vocab_counter.update(tokens)

    print(i)
    print('_ number of tokens', len(vocab_counter))
    frequent_tokens = [k for k, v in vocab_counter.items() if v >= min_occur]
    print(
        f'- number of tokens with min {min_occur} occurrences',
        len(frequent_tokens))

    epi = epitran.Epitran(ortho_name)

    vocab_all = []
    print('Converting to IPA...')
    for token in tqdm(frequent_tokens):
        # doing ''.join(ipa_segs(s)) removes non-ipa characters from the string
        segments = ft.ipa_segs(epi.transliterate(token))
        vocab.update(segments)
        if segments:
            vocab_all.append((
                "",
                ''.join(segments),
                lang,
                ""
            ))

    vocab_all = set(vocab_all)
    print('- number of tokens after cleaning up', len(vocab_all))
    with open(f"data/ipa_tokens_{lang}.txt", 'w') as f:
        f.write('\n'.join(vocab_all))

    with open(f"data/multi.tsv", 'a') as f:
        f.write('\n'.join(["\t".join(x) for x in vocab_all]))

    with open(f"data/vocab_{lang}.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab))))

    print(f'Saved {lang}')


def process_ipa_cmu_dict():
    cmu_pronunciation = [
        x.rstrip("\n").split("  ")
        for x in open('data/raw/cmudict-0.7b.txt') if x[0].isalpha()
    ]
    cmu_pronunciation = {x[0]: x[1] for x in cmu_pronunciation}

    vocab_all = []
    vocab_chars = set()
    print('Gathering tokens for en...')
    with open('data/raw/cmudict-0.7b-ipa.txt') as f:
        for line in tqdm(f):
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
    vocab_all = set(vocab_all)

    print('   number of tokens after cleaning up', len(vocab_all))
    with open(f"data/ipa_tokens_en.txt", 'w') as f:
        f.write('\n'.join([x[1] for x in vocab_all]))

    with open(f"data/multi.tsv", 'a') as f:
        f.write('\n'.join(["\t".join(x) for x in vocab_all]))

    with open(f"data/vocab_en.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_chars))))

    print(f'Saved en')


if __name__ == '__main__':
    # clear multi file
    open(f"data/multi.tsv", 'w').close()

    # for english, we use the IPA version of the CMU pronunciation dict from
    # https://github.com/menelik3/cmudict-ipa
    process_ipa_cmu_dict()

    LANG_TO_ORTHO = {
        'am': 'amh-Ethi',
        'bn': 'ben-Beng',
        'uz': 'uzb-Latn',
        'pl': 'pol-Latn',
        'es': 'spa-Latn',
        'sw': 'swa-Latn',
    }
    for lang, ortho_name in LANG_TO_ORTHO.items():
        transliterate_lang(lang, ortho_name)

    # consolidate vocab
    vocab_multi = set()
    langs = ['en'] + list(LANG_TO_ORTHO.keys())
    for lang in langs:
        with open(f"data/vocab_{lang}.txt") as f:
            vocab_multi.update(f.read().split())

    with open(f"data/vocab_multi.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab_multi))))

    print("Multi vocab file generated")
