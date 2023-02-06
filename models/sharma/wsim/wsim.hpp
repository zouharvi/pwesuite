#include <algorithm>
#include <cmath>
#include <bitset>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define DEBUG 1

namespace wsim {

using feature_type = uint32_t;
using phone_list = std::vector<feature_type>;

const int BIGRAM = 1;
const int INSERT_BEG_END = 2;
const int VOWEL_BUFF = 4;
const int RHYME_BUFF = 8;

class Dictionary {
  public:
    Dictionary(const std::string &path);
    ~Dictionary();

    std::string getPath() const;

    int size() const;
    int getIndex(const std::string &word) const;
    std::string getWord(int idx) const;

    float score(int w1, int w2, int flags, float nonDiagonalPenalty) const;

    std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<float>>
    randomScores(int n, int flags, float nonDiagonalPenalty) const;

    // float checkEmbeddings(const std::vector<std::vector<float>>&, int flags, float nonDiagonalPenalty) const;

    std::vector<std::pair<float, std::string>>
    topSimilar(const std::string &word, int k, int flags,
               float nonDiagonalPenalty) const;

  protected:
    std::string m_path;
    std::vector<phone_list> m_phones;
    std::vector<std::string> m_words;
    std::unordered_map<std::string, int> m_words_idx;
    float score_unigram(int w1, int w2, int flags,
                        float nonDiagonalPenalty) const;
    float score_bigram(int w1, int w2, int flags,
                       float nonDiagonalPenalty) const;
};
} // namespace wsim
