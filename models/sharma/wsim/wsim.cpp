#include "wsim.hpp"

namespace wsim {

const std::map<std::string, std::vector<std::string>> PHONE_FEATURES = {
    {"^", {"beg"}},
    {"$", {"end"}},
    {"M", {"blb", "nas"}},
    {"P", {"vls", "blb", "stp"}},
    {"B", {"vcd", "blb", "stp"}},
    {"F", {"vls", "lbd", "frc"}},
    {"V", {"vcd", "lbd", "frc"}},
    {"TH", {"vls", "dnt", "frc"}},
    {"DH", {"vcd", "dnt", "frc"}},
    {"N", {"alv", "nas"}},
    {"T", {"vls", "alv", "stp"}},
    {"D", {"vcd", "alv", "stp"}},
    {"S", {"vls", "alv", "frc"}},
    {"Z", {"vcd", "alv", "frc"}},
    {"R", {"alv", "apr"}},
    {"L", {"alv", "lat"}},
    {"SH", {"vls", "pla", "frc"}},
    {"ZH", {"vcd", "pla", "frc"}},
    {"Y", {"pal", "apr"}},
    {"NG", {"vel", "nas"}},
    {"K", {"vls", "vel", "stp"}},
    {"G", {"vcd", "vel", "stp"}},
    {"W", {"lbv", "apr"}},
    {"HH", {"glt", "apr"}},
    {"CH", {"vls", "alv", "stp", "frc"}},
    {"JH", {"vcd", "alv", "stp", "frc"}},
    {"AO", {"lmd", "bck", "rnd", "vwl"}},
    {"AA", {"low", "bck", "unr", "vwl"}},
    {"IY", {"hgh", "fnt", "unr", "vwl"}},
    {"UW", {"hgh", "bck", "rnd", "vwl"}},
    {"EH", {"lmd", "fnt", "unr", "vwl"}},
    {"IH", {"smh", "fnt", "unr", "vwl"}},
    {"UH", {"smh", "bck", "rnd", "vwl"}},
    {"AH", {"mid", "cnt", "unr", "vwl"}},
    {"AE", {"low", "fnt", "unr", "vwl"}},
    {"EY", {"lmd", "smh", "fnt", "unr", "vwl"}},
    {"AY", {"low", "smh", "fnt", "cnt", "unr", "vwl"}},
    {"OW", {"umd", "smh", "bck", "rnd", "vwl"}},
    {"AW", {"low", "smh", "bck", "cnt", "unr", "rnd", "vwl"}},
    {"OY", {"lmd", "smh", "bck", "fnt", "rnd", "unr", "vwl"}},
    {"ER", {"umd", "cnt", "rzd", "vwl"}}};

const feature_type START_BIT = 0;
const feature_type END_BIT = 1;
const feature_type VOWEL_BIT = 2;
const feature_type VOWEL_FLAG = 1 << VOWEL_BIT;

const auto FEATURES_MAP = []() {
    std::unordered_map<std::string, int> features = {
        {"beg", START_BIT}, {"end", END_BIT}, {"vwl", VOWEL_BIT}};
    std::unordered_map<std::string, feature_type> feature_map;
    for (auto &pair : PHONE_FEATURES) {
        feature_type feature_bits = 0;
        for (auto &feature : pair.second) {
            auto result = features.find(feature);
            if (result == features.end()) {
                features.insert({feature, (int)features.size()});
            }
            feature_bits |= (1 << features[feature]);
        }
        feature_map[pair.first] = feature_bits;
    }
#ifdef DEBUG
    std::cout << "Total phones = " << feature_map.size() - 2 << " + 2" << '\n';
    std::cout << "Total features = " << features.size() << '\n';
#endif
    return feature_map;
}();

const auto REV_FEATURES_MAP = []() {
    std::unordered_map<feature_type, std::string> m;
    for (const auto &pair : FEATURES_MAP) {
        m.insert({pair.second, pair.first});
    }
    return m;
}();

feature_type getFeature(const std::string &phone) {
    std::string p;
    p.reserve(phone.size());
    for (char c : phone) {
        if ('a' <= c && c <= 'z')
            p.push_back(c - 32);
        else if ('A' <= c && c <= 'Z')
            p.push_back(c);
    }
    if(FEATURES_MAP.find(p) == FEATURES_MAP.end()){
        std::cout << "WARNING!!! FEATURE NOT FOUND '" << p << "'" << std::endl;
    }
    return FEATURES_MAP.at(p);
}

phone_list addBegEnd(const phone_list &phones) {
    phone_list list;
    list.reserve(phones.size() + 2);
    list.push_back(FEATURES_MAP.at("^"));
    for (const auto phone : phones)
        list.push_back(phone);
    list.push_back(FEATURES_MAP.at("$"));
    // for(auto c: list){
    //     std::cout << c << ';';
    // }
    // std::cout << std::endl;
    return list;
}

inline uint32_t bitCount(const uint32_t v) {
#ifdef _MSC_VER
    return _mm_popcnt_u32(v);
#else
    return __builtin_popcount(v);
#endif
}

inline float getSimilarity(const feature_type f1, const feature_type f2) {
    return bitCount(f1 & f2) / (float)(bitCount(f1 | f2));
}

inline float getSimilarity(const feature_type a1, const feature_type a2,
                           const feature_type b1, const feature_type b2,
                           const bool vowel_buff) {
    const feature_type w1 = a1 | a2;
    const feature_type w2 = b1 | b2;
    float score = bitCount(w1 & w2) / (float)(bitCount(w1 | w2));
    if (vowel_buff)
        score = ((a2 & VOWEL_FLAG) && (b2 & VOWEL_FLAG) && a2 == b2)
                    ? sqrt(score)
                    : (score * score);
    return score;
}

Dictionary::Dictionary(const std::string &path) : m_path(path) {
#ifdef DEBUG
    std::cout << "Dictionary path = " << path << '\n';
#endif
    std::ifstream fin(path);
    for (std::string line; std::getline(fin, line);) {
        if (!line.size() || line[0] == ';')
            continue;
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        auto result = m_words_idx.find(word);
        if (result == m_words_idx.end()) {
            m_words.push_back(word);
            m_words_idx[word] = m_phones.size();
            phone_list phones;
            while (iss >> word)
                phones.push_back(getFeature(word));
            m_phones.push_back(std::move(phones));
        }
    }
#ifdef DEBUG
    std::cout << "Dictionary size = " << m_phones.size() << '\n';
#endif
    // const auto a1 = getFeature("W");
    // std::cout << "W = " << std::bitset<31>(a1) << '\n';
    // const auto a2 = getFeature("AH");
    // std::cout << "AH = " << std::bitset<31>(a2) << '\n';
    // const auto b1 = FEATURES_MAP.at("^");
    // std::cout << "^ = " << std::bitset<31>(b1) << '\n';
    // const auto b2 = getFeature("AH");
    // std::cout << "Similarity(W+AH, ^+AH) = " << getSimilarity(a1, a2, b1, b2,
    // true) << '\n';
}

Dictionary::~Dictionary() {
#ifdef DEBUG
    std::cout << "Dictionary " << m_path << " Unloaded \n";
#endif
}

std::string Dictionary::getPath() const { return m_path; }

int Dictionary::size() const { return m_words.size(); }

int Dictionary::getIndex(const std::string &word) const {
    auto result = m_words_idx.find(word);
    if (result == m_words_idx.end())
        return -1;
    return result->second;
}

std::string Dictionary::getWord(int idx) const { return m_words[idx]; }

float Dictionary::score_unigram(int w1, int w2, int flags,
                                float nonDiagonalPenalty) const {
    auto p1 = (flags & INSERT_BEG_END) ? addBegEnd(m_phones[w1]) : m_phones[w1];
    auto p2 = (flags & INSERT_BEG_END) ? addBegEnd(m_phones[w2]) : m_phones[w2];
    const int n1 = p1.size();
    const int n2 = p2.size();
    std::vector<float> even(n1), odd(n1);
    even[0] = getSimilarity(p2[0], p1[0]);
    for (int i = 1; i < n1; i++)
        even[i] = getSimilarity(p2[0], p1[i]) + even[i - 1];
    // for (auto &v : even)
    //     printf("%1.5f  ", v);
    // std::cout << '\n';
    for (int i = 1; i < n2; i++) {
        auto &cur = (i & 1) ? odd : even;
        auto &prev = (i & 1) ? even : odd;
        // nonDiagonalPenalty is avoided because of no choice
        cur[0] = getSimilarity(p2[i], p1[0]) + prev[0];
        for (int j = 1; j < n1; j++) {
            const auto sim = getSimilarity(p2[i], p1[j]);
            if (p2[i] == p1[j]) {
                cur[j] = sim + prev[j - 1];
            } else {
                if (prev[j] > cur[j - 1])
                    cur[j] = sim * nonDiagonalPenalty + prev[j];
                else
                    cur[j] = sim * nonDiagonalPenalty + cur[j - 1];
            }
        }
        // for (auto &v : cur)
        //     printf("%1.5f  ", v);
        // std::cout << '\n';
    }
    const int base = (flags & INSERT_BEG_END) ? 2 : 0;
    float best = (((n2 & 1) ? even : odd).back() - base);
    return best / (float)(std::max(n1, n2) - base);
}

float Dictionary::score_bigram(int w1, int w2, int flags,
                               float nonDiagonalPenalty) const {
    auto p1 = (flags & INSERT_BEG_END) ? addBegEnd(m_phones[w1]) : m_phones[w1];
    auto p2 = (flags & INSERT_BEG_END) ? addBegEnd(m_phones[w2]) : m_phones[w2];
    const bool vowel_buff = (flags & VOWEL_BUFF);
    const int n1 = p1.size() - 1;
    const int n2 = p2.size() - 1;
    if (n1 < 1 || n2 < 1)
        return 0;
    // for (int j = 0; j <= n1; j++)
    //     std::cout << REV_FEATURES_MAP.at(p1[j]) << ' ';
    // std::cout << '\n';
    // for (int j = 0; j <= n2; j++)
    //     std::cout << REV_FEATURES_MAP.at(p2[j]) << ' ';
    // std::cout << '\n';
    // for (int i = 0; i < n2; i++) {
    //     for (int j = 0; j < n1; j++) {
    //         const auto v =
    //             getSimilarity(p2[i], p2[i + 1], p1[j], p1[j + 1],
    //             vowel_buff);
    //         printf("%1.5f  ", v);
    //     }
    //     std::cout << '\n';
    // }
    // std::cout << '\n';
    std::vector<float> even(n1), odd(n1);
    even[0] = getSimilarity(p2[0], p2[1], p1[0], p1[1], vowel_buff);
    for (int i = 1; i < n1; i++)
        even[i] = getSimilarity(p2[0], p2[1], p1[i], p1[i + 1], vowel_buff) +
                  even[i - 1];
    // for (auto &v : even)
    //     printf("%1.5f  ", v);
    // std::cout << '\n';
    for (int i = 1; i < n2; i++) {
        auto &cur = (i & 1) ? odd : even;
        auto &prev = (i & 1) ? even : odd;
        // nonDiagonalPenalty is avoided because of no choice
        cur[0] =
            getSimilarity(p2[i], p2[i + 1], p1[0], p1[1], vowel_buff) + prev[0];
        for (int j = 1; j < n1; j++) {
            const auto sim =
                getSimilarity(p2[i], p2[i + 1], p1[j], p1[j + 1], vowel_buff);
            if (sim < 1) {
                if (prev[j] > cur[j - 1])
                    cur[j] = sim * nonDiagonalPenalty + prev[j];
                else
                    cur[j] = sim * nonDiagonalPenalty + cur[j - 1];
            } else {
                cur[j] = sim + prev[j - 1];
            }
        }
        // for (auto &v : cur)
        //     printf("%1.5f  ", v);
        // std::cout << '\n';
    }
    float score = ((n2 & 1) ? even : odd).back() / std::max(n1, n2);
    return score;
}

float Dictionary::score(int w1, int w2, int flags,
                        float nonDiagonalPenalty) const {
    if (flags & BIGRAM)
        return score_bigram(w1, w2, flags, nonDiagonalPenalty);
    return score_unigram(w1, w2, flags, nonDiagonalPenalty);
}

#define FAST_TRAIN

#ifdef FAST_TRAIN
const std::vector<std::vector<int>> dataset = {
    {110591, 127833, 103577, 63994, 37558, 118189, 118276, 33874, 118652, 121536, 77476, 58064, 41187, 20796, 16762, 70539, 112328, 120747, 120107, 79105, 60812, 105141, 108262, 109860, 92006, 55044},
    { 92239, 106825, 14871, 26831, 128685, 52332, 70035, 34429, 111397, 12033, 88873, 49269, 26144, 43173, 12361, 91563, 114005, 92523, 113310, 93846, 49401, 133852, 133853, 92237, 133854, 111008},
    {131504, 109872, 121539, 70753, 97089, 97405, 133858, 15812, 26092, 130766, 76996, 23037, 51871, 16795, 28113, 130408, 16241, 119331, 130885, 129168, 131044, 133855, 133856, 128217, 116742, 133857},
    { 99481, 47152, 6278, 30967, 54313, 40783, 116705, 36127, 9621, 101345, 68786, 76475, 124263, 88150, 67835, 103748, 100004, 59149, 37300, 100477, 66845, 101050, 99472, 70723, 97575, 133859}
};
#endif

std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<float>>
Dictionary::randomScores(int n, int flags, float nonDiagonalPenalty) const {
    std::vector<int> i1(n), i2(n);
    std::vector<float> s(n);
#ifdef FAST_TRAIN
    const auto& ds = dataset[rand() % 4];
    i1[0] = ds[0];
    i2[0] = ds[1 + (rand() % (ds.size() - 1))];
    s[0] = score(i1[0], i2[0], flags, nonDiagonalPenalty);
    for (int i = 1; i < n; i++) {
#else
    for (int i = 0; i < n; i++) {
#endif
        i1[i] = rand() % m_words.size();
        i2[i] = rand() % m_words.size();
        s[i] = score(i1[i], i2[i], flags, nonDiagonalPenalty);
    }
    return {{i1, i2}, s};
}

std::vector<std::pair<float, std::string>>
Dictionary::topSimilar(const std::string &word, int k, int flags,
                       float nonDiagonalPenalty) const {
    // std::cout << "flags: " << flags << std::endl;
    const auto idx = getIndex(word);
    std::vector<std::pair<float, std::string>> vals;
    vals.reserve(m_words_idx.size());
    for (auto &wi_pair : m_words_idx)
        vals.push_back({score(idx, wi_pair.second, flags, nonDiagonalPenalty),
                        wi_pair.first});
    std::sort(vals.begin(), vals.end(),
              [](const std::pair<float, std::string> &e1,
                 const std::pair<float, std::string> &e2) {
                  return e1.first > e2.first;
              });
    if (0 < k && k < (int)vals.size())
        vals.resize(k);
    return vals;
}
} // namespace wsim

int main(){
    using namespace wsim;
    Dictionary d("../data/cmudict-0.7b-with-vitz-nonce");
    for(auto res: d.topSimilar("SIT", 10, INSERT_BEG_END | BIGRAM | VOWEL_BUFF, 0.4)){
        std::cout << res.first << " | " << res.second << '\n';
    }
}
