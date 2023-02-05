# taken from
# https://github.com/aparrish/phonetic-similarity-vectors
# https://github.com/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition/blob/main/Fine_tuning_XLSR_Wav2Vec2_for_Phoneme_Recognition.ipynb

from main.utils import UNK_SYMBOL


class IPA2ARP:
    ipa2arp = {
        'ɑ': 'aa', 'æ': 'ae', 'ʌ': 'ah', 'ɔ': 'ao', 'aʊ': 'aw', 'ə': 'ax', 'ɚ': 'axr', 'aɪ': 'ay', 'ɛ': 'eh', 'ɝ': 'er', 'eɪ': 'ey', 'ɪ': 'ih', 'ɨ': 'ix',
        'i': 'iy', 'oʊ': 'ow', 'ɔɪ': 'oy', 'ʊ': 'uh', 'u': 'uw', 'ʉ': 'ux', 'b': 'b', 'tʃ': 'ch', 'd': 'd', 'ð': 'dh', 'ɾ': 'dx', 'l̩': 'el', 'm̩': 'em',
        'n̩': 'en', 'f': 'f', 'ɡ': 'g', 'h': 'h', 'dʒ': 'jh', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'ŋ': 'ng', 'ɾ̃': 'nx', 'p': 'p', 'ʔ': 'q', 'ɹ': 'r',
        's': 's', 'ʃ': 'sh', 't': 't', 'θ': 'th', 'v': 'v', 'w': 'w', 'ʍ': 'wh', 'j': 'y', 'z': 'z', 'ʒ': 'zh', ':': ':', 'W': 'aw', 'Y': 'ay', 'e': 'ey',
        'o': 'ow', 'O': 'oy', 'C': 'ch', 'g': 'g', 'J': 'jh', 'ə̥': 'ax-h', 'b̚': 'bcl', 'd̚': 'dcl', 'ŋ̍': 'eng', 'ɡ̚': 'gcl', 'ɦ': 'hv', 'k̚': 'kcl',
        'p̚': 'pcl', 't̚': 'tcl', 'S': 'epi', 'P': 'pau'
    }

    def __init__(self):
        # sort by first key being the longest
        self.ipa2arp = sorted(
            self.ipa2arp.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self.ipa2arp = {k: v.upper() for k, v in self.ipa2arp}

    def convert(self, text):
        cmu = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i:i + 2] in self.ipa2arp.keys():
                cmu.append(self.ipa2arp[text[i:i + 2]])
                i += 1
            elif text[i] in self.ipa2arp.keys():
                cmu.append(self.ipa2arp[text[i]])
            else:
                cmu.append(UNK_SYMBOL)
                # skip phonemes that can't be described
                # cmu.append(text[i])
            i += 1

        return cmu
