"""
Perceptron Algorithm - estimates weight vector for part of speech tagging.

Author: Hyonjee Joo

Usage: python perceptron.py data/tag_train.dat > tagger.model
"""

import sys
import pprint
from subprocess import PIPE
import subprocess
from collections import defaultdict

ITERATIONS = 5

class Sentence(object):
    def __init__(self, words, tags, bigrams, histories):
        self.words = words
        self.tags = tags
        self.bigrams = bigrams
        self.histories = histories
    
    def __repr__(self):
        obj = {'words': self.words, 'tags': self.tags,
               'bigrams': self.bigrams, 'hist': self.histories}
        return str(obj)


"""
methods for calling scripts from within code
"""
def process(args):
    return subprocess.Popen(args, stdin=PIPE, stdout=PIPE)

def call(process, stdin):
    output = process.stdin.write(stdin + "\n\n")
    line = ""
    while 1:
        l = process.stdout.readline()
        if not l.strip(): break
        line += l
    return line


"""
Perceptron algorithm. Returns the weight vector, v.
"""
def run_perceptron(sentences, f):
    v = defaultdict(float)
    for t in range(ITERATIONS):
        for s in sentences:
            z_tags = arg_max(s.words, s.histories, f, v)
            if z_tags != s.tags:
                update(v, s, z_tags, f)
    return v


"""
Compute arg max tags for given sentence and histories.
"""
DECODE_SERVER = process(["python", "lib/tagger_decoder.py", "HISTORY"])
def arg_max(words, histories, f, v):
    # calculate scores
    scored_histories = []
    for i, word in enumerate(words, start=1):
        for t0, t1 in histories[i]:
            score = 0.0

            tag_feature = ":".join(["TAG", word, t1])
            score += v[tag_feature] * f[tag_feature]

            bigram_feature = ":".join(["BIGRAM", t0, t1])
            score += v[bigram_feature] * f[bigram_feature]

            th_feature = ":".join(["TH", word, t1])
            score += v[th_feature] * f[th_feature]

            cap_feature = ":".join(["CAP", t1])
            score += v[cap_feature] * f[cap_feature]

            len_feature = ":".join(["LEN", str(len(word)), t1])
            score += v[len_feature] * f[len_feature]

            for j in range(1, 4):
                if len(word) >= j:
                    suff_feature = ":".join(["SUFF", word[-j:], str(j), t1])
                    score += v[suff_feature] * f[suff_feature]
                else:
                    break

            scored_histories.append(" ".join([str(i), t0, t1, str(score)]))

    arg_max_bigrams = call(DECODE_SERVER, "\n".join(scored_histories))
    bigrams = arg_max_bigrams.strip().split("\n")
    bigrams.pop() # remove STOP tag at end

    arg_max_tags = []
    for bigram in bigrams:
        i, t0, t1 = bigram.strip().split()
        arg_max_tags.append(t1)
    return arg_max_tags

"""
Updates weight vector in perceptron algorithm.
"""
def update(v, s, z_tags, f):
    z_bigrams = [("*", z_tags[0])]
    for i, z_tag in enumerate(z_tags[1:]):
        z_bigrams.append((z_tags[i], z_tag))

    for word, y_tag, y_bigram, z_tag, z_bigram in zip(s.words, s.tags, s.bigrams, z_tags,
            z_bigrams):
        y_tag_feat = ":".join(["TAG", word, y_tag])
        v[y_tag_feat] += f[y_tag_feat]
        z_tag_feat = ":".join(["TAG", word, z_tag])
        v[z_tag_feat] -= f[z_tag_feat]

        y_bigram_feat = ":".join(["BIGRAM", y_bigram[0], y_bigram[1]])
        v[y_bigram_feat] += f[y_bigram_feat]
        z_bigram_feat = ":".join(["BIGRAM", z_bigram[0], z_bigram[1]])
        v[z_bigram_feat] -= f[z_bigram_feat]

        y_th_feat = ":".join(["TH", word, y_tag])
        v[y_th_feat] += f[y_th_feat]
        z_th_feat = ":".join(["TH", word, z_tag])
        v[z_th_feat] -= f[z_th_feat]
        
        y_cap_feat = ":".join(["CAP", y_tag])
        v[y_cap_feat] += f[y_cap_feat]
        z_cap_feat = ":".join(["CAP", z_tag])
        v[z_cap_feat] -= f[z_cap_feat]

        y_len_feat = ":".join(["LEN", str(len(word)), y_tag])
        v[y_len_feat] += f[y_len_feat]
        z_len_feat = ":".join(["LEN", str(len(word)), z_tag])
        v[z_len_feat] -= f[z_len_feat]

        for j in range(1,4):
            if len(word) >= j:
                y_suff_feat = ":".join(["SUFF", word[-j:], str(j), y_tag])
                v[y_suff_feat] += f[y_suff_feat]
                z_suff_feat = ":".join(["SUFF", word[-j:], str(j), z_tag])
                v[z_suff_feat] -= f[z_suff_feat]

"""
Methods to process training data.
"""
def get_histories(train_sentences):
    gold_server = process(["python", "lib/tagger_history_generator.py", "GOLD"])
    enum_server = process(["python", "lib/tagger_history_generator.py", "ENUM"])

    gold_histories = ""
    all_histories= ""
    for train_sentence in train_sentences:
        gold_histories += call(gold_server, train_sentence) + "\n"
        all_histories += call(enum_server, train_sentence) + "\n"
    return (gold_histories, all_histories)

def pre_process_train_dat(train_file_name):
    train_file = open(train_file_name, 'r')
    train_dat = train_file.read()
    train_file.close()

    train_sentences = train_dat.strip().split("\n\n")
    sentences = []
    sentence_tags = []
    for s in train_sentences:
        word_tags = s.strip().split("\n")
        words = []
        tags = []
        for word_tag in word_tags:
            word, tag = word_tag.split("\t")
            words.append(word)
            tags.append(tag)
        sentences.append(words)
        sentence_tags.append(tags)

    # get histories and turn to arrays of histories per sentence
    gold_histories, all_histories = get_histories(train_sentences) 
    sentence_bigrams = []
    for s in gold_histories.strip().split("\n\n"):
        s_bigrams = s.strip().split("\n")
        temp = []
        for b in s_bigrams:
            i, t0, t1 = b.split()
            temp.append((t0, t1))
        sentence_bigrams.append(temp)

    sentence_histories = []
    for s in all_histories.strip().split("\n\n"):
        s_histories = s.strip().split("\n")
        temp = defaultdict(list)
        for h in s_histories:
            i, t0, t1 = h.split()
            temp[int(i)].append((t0, t1))
        sentence_histories.append(temp)

    assert len(sentences) == len(sentence_tags) == \
            len(sentence_bigrams) == len(sentence_histories)

    # make sentences
    sentence_objects = []
    for sentence, tags, bigrams, histories in zip(sentences, sentence_tags,
            sentence_bigrams, sentence_histories):
        sentence_objects.append(Sentence(sentence, tags, bigrams, histories))
    return sentence_objects

def init_features(sentences):
    feature_dict = defaultdict(int)
    for sentence in sentences:
        for word, tag in zip(sentence.words, sentence.tags):
            tag_feature = ":".join(["TAG", word, tag])
            feature_dict[tag_feature] = 1
            # TH feature
            if len(word) > 1 and word[:2] == "Th" and tag == "DET":
                th_feature = ":".join(["TH", word, "DET"])
                feature_dict[th_feature] =  1
            # CAP feature
            if word[:1].isupper():
                cap_feature = ":".join(["CAP", tag])
                feature_dict[cap_feature] = 1
            # LEN feature
            len_feature = ":".join(["LEN", str(len(word)), tag])
            feature_dict[len_feature] = 1
            # SUFF feature
            for j in range(1, 4):
                if len(word) >= j:
                    suff_feature = ":".join(["SUFF", word[-j:], str(j), tag])
                    feature_dict[suff_feature] = 1
                else:
                    break
        for t0, t1 in sentence.bigrams:
            bigram_feature = ":".join(["BIGRAM", t0, t1])
            feature_dict[bigram_feature] = 1
    return feature_dict


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python perceptron.py tag_train.dat\n")
        sys.exit(1)

    sentences = pre_process_train_dat(sys.argv[1])
    features = init_features(sentences)

    weight_vector = run_perceptron(sentences, features) 

    # write out suffix model
    model = ""
    for feature in weight_vector:
        if weight_vector[feature] != 0:
            model += feature + " " + str(weight_vector[feature]) + "\n"
    print(model.strip())
