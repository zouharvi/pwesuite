# taken from
# https://github.com/aparrish/phonetic-similarity-vectors

class IPA2CMU:
    def __init__(self):
        # define the phone mapping in ipa2cmu.map and read this into a dictionary 
        self.ipa2cmu = dict([tuple(f.strip().split('\t')) for f in open('data/ipa2cmu.map')]) 
        self.ipa2cmu = {k:v.upper() for k,v in self.ipa2cmu.items()}

    def convert(self, text):
        cmu = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i:i+2] in self.ipa2cmu.keys():
                cmu.append(self.ipa2cmu[text[i:i+2]])
                i += 1
            elif text[i] in self.ipa2cmu.keys():
                cmu.append(self.ipa2cmu[text[i]])
            else:
                pass
                # skip phonemes that can't be described
                # cmu.append(text[i])
            i += 1

        return cmu