# distutils: language = c++
# distutils: sources = wsim.cpp

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "wsim.hpp" namespace "wsim":
    cdef cppclass Dictionary:
        Dictionary(string)
        string getPath()
        int size()
        int getIndex(string word)
        string getWord(int idx)
        float score(int, int, int, float)
        pair[pair[vector[int], vector[int]], vector[float]] randomScores(int, int, float)
        vector[pair[float,string]] topSimilar(string, int, int, float)

cdef class wsimdict:
    BIGRAM = 1
    INSERT_BEG_END = 2
    VOWEL_BUFF = 4
    RHYME_BUFF = 8
    DECODING = 'utf-8'

    cdef Dictionary* ptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, path):
        self.ptr = new Dictionary(path.encode())
    def __dealloc__(self):
        del self.ptr
    def __len__(self):
        return self.ptr.size()
    def get_path(self):
        return self.ptr.getPath().decode(self.DECODING)
    def get_index(self, word):
        return self.ptr.getIndex(word.encode())
    def get_word(self, idx):
        return self.ptr.getWord(idx).decode(self.DECODING)
    def score(self, w1, w2, flags=0, non_diagonal_penalty=1):
        return self.ptr.score(w1, w2, flags, non_diagonal_penalty)
    def random_scores(self, n, flags=0, non_diagonal_penalty=1):
        return self.ptr.randomScores(n, flags, non_diagonal_penalty)
    def similarity(self, w1, w2, flags=0, non_diagonal_penalty=1):
        idx1 = self.get_index(w1)
        idx2 = self.get_index(w2)
        return self.score(idx1, idx2, flags, non_diagonal_penalty)
    def top_similar(self, word, k, flags=0, non_diagonal_penalty=1):
        return self.ptr.topSimilar(word.encode(), k, flags, non_diagonal_penalty)
    def __str__(self):
        return '<Dictionay object from "' + self.get_path() + '">'
        # from wsim import wsimdict as wd
        # a = wd('../phonetic-similarity-vectors/cmudict-0.7b-with-vitz-nonce')
