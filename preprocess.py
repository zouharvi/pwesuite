from collections import Counter

import epitran, panphon
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
            if i > 1e7: break
            tokens = line.decode('utf-8').split()
            vocab_counter.update(tokens)
    print(i)
    print('   number of tokens', len(vocab_counter))
    frequent_tokens = [k for k, v in vocab_counter.items() if v >= min_occur]
    print(f'   number of tokens with min {min_occur} occurrences', len(frequent_tokens))

    epi = epitran.Epitran(ortho_name)

    ipas = []
    print('Converting to IPA...')
    for token in tqdm(frequent_tokens):
        # doing ''.join(ipa_segs(s)) removes non-ipa characters from the string
        segments = ft.ipa_segs(epi.transliterate(token))
        vocab.update(segments)
        if segments:
            ipas.append(''.join(segments))

    ipa_set = set(ipas)
    print('   number of tokens after cleaning up', len(ipa_set))
    with open(f"data/ipa_tokens_{lang}.txt", 'w') as f:
        f.write('\n'.join(ipa_set))

    print('Saved to file!')

    with open(f"data/vocab_{lang}.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab))))


def process_ipa_cmu_dict():
    fpath = f'data/raw/cmudict-0.7b-ipa.txt'
    ipas = []
    vocab = set()
    print('Gathering tokens for en...')
    with open(fpath) as f:
        for line in tqdm(f):
            if (ord('A') <= ord(line[0]) <= ord('Z')):
                ipa = line.split('\t')[1]
                ipa = ipa.split(' ')[0]  # if multiple pronunciations, take first one
                ipa = re.sub('[,ˌˈ]', '', ipa)  # remove stress marks and commas
                segments = ft.ipa_segs(ipa)
                vocab.update(segments)
                if segments:
                    ipas.append(''.join(segments))
    ipa_set = set(ipas)
    print('   number of tokens after cleaning up', len(ipa_set))
    with open(f"data/ipa_tokens_en.txt", 'w') as f:
        f.write('\n'.join(ipa_set))

    print('Saved to file!')

    with open(f"data/vocab_en.txt", 'w') as f:
        f.write('\n'.join(sorted(list(vocab))))

if __name__ == '__main__':

    # for english, we use the IPA version of the CMU pronunciation dict from
    # https://github.com/menelik3/cmudict-ipa
    process_ipa_cmu_dict()
    #
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
    master_vocab = set()
    langs = ['en'] + list(LANG_TO_ORTHO.keys())
    for lang in langs:
        with open(f"data/vocab_{lang}.txt") as f:
            master_vocab.update(f.read().split())

    with open(f"data/vocab_{'_'.join(langs)}", 'w') as f:
        f.write('\n'.join(sorted(list(master_vocab))))
    print("Master vocab file generated")