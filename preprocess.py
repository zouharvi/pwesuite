from collections import Counter

import epitran, panphon
import re
from tqdm import tqdm


# https://www.w3.org/TR/elreq/#ethiopic_punctuation
amharic_symb = '፠፡።፣፤፥፦፧፨‘’“”‹›«»€…'
# https://en.wikipedia.org/wiki/Bengali_alphabet#Punctuation_marks, https://en.wikipedia.org/wiki/Bengali_numerals
bengali_symb = '০১২৩৪৫৬৭৮৯৹৷৶৴৵৸₹–।'
english_punc = r'!"#$%&\'\(\)*\+,-./:;<=>?@\[\\\]^_`{|}~'
other_punc = r'‌'
punctuations = english_punc + amharic_symb + bengali_symb

ft = panphon.FeatureTable()
LANG_TO_ORTHO = {
    'am': 'amh-Ethi',
    'bn': 'ben-Beng',
}
MIN_OCCUR = 5

vocab = set()

for lang, ortho_name in LANG_TO_ORTHO.items():
    print(f'=== Processing language: {lang} ===')
    vocab_counter = Counter()
    fpath = f'data/raw/{lang}.txt'
    print('Gathering tokens...')
    num_lines = sum(1 for line in open(fpath))
    with open(fpath) as f:
        for line in tqdm(f, total=num_lines):
            tokens = re.sub(f'[a-zA-Z0-9{punctuations}]', ' ', line).split()
            vocab_counter.update(tokens)
    print('   number of tokens', len(vocab_counter))
    frequent_tokens = [k for k, v in vocab_counter.items() if v >= MIN_OCCUR]
    print(f'   number of tokens with min {MIN_OCCUR} occurrences', len(frequent_tokens))

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

with open(f"data/vocab.txt", 'w') as f:
    f.write('\n'.join(vocab))
print("Vocab file generated")