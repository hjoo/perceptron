"""
Part of speech tagger that uses suffix features and some custom features
defined in README.

Author: Hyonjee Joo (hj2339)

Usage: python tagger.py tagger.model data/tag_dev.dat > tag_dev.out
"""

import sys
import pprint
from subprocess import PIPE
import subprocess
from collections import defaultdict

def process(args):
    return subprocess.Popen(args, stdin=PIPE, stdout=PIPE)

def call(process, stdin):
    output = process.stdin.write(stdin + "\n\n")
    line = ""
    while 1:
        l = process.stdout.readline()
        if not l.strip(): break
        line +=l
    return line

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(
                "Usage: python new_tagger.py new_tagger.model tag_dev.dat\n")
        sys.exit(1)

    # weight vector, key = feature, value = weight
    v = defaultdict(float)
    with open(sys.argv[1], 'r') as suffix_model_file:
        for line in suffix_model_file:
            feat_weight = line.strip().split(" ")
            v[feat_weight[0]] = float(feat_weight[1])

    # create history server
    enum_server = process(["python", "lib/tagger_history_generator.py", "ENUM"])
    # create decoder server
    decoder_server = process(["python", "lib/tagger_decoder.py", "HISTORY"])

    # read in dev sentences
    dev_file = open(sys.argv[2], 'r')
    dev_dat = dev_file.read()
    dev_sentences = dev_dat.strip().split("\n\n") # no new line at end of sentences

    for sentence in dev_sentences:
        # enumerate all possible histories
        words = sentence.split("\n")
        histories = call(enum_server, sentence)
        histories = histories.strip().split("\n")

        # score histories
        scored_histories = ""
        for history in histories:
            i, t0, t1 = history.strip().split()
            i = int(i)

            bigram_feature = ":".join(["BIGRAM", t0, t1]) 
            tag_feature = ":".join(["TAG", words[i-1], t1])
            th_feature = ":".join(["TH", words[i-1], t1])
            cap_feature = ":".join(["CAP", t1])
            len_feature = ":".join(["LEN", str(len(words[i-1])), t1])

            history_weight = 0.0
            history_weight += v[bigram_feature]
            history_weight += v[tag_feature]
            history_weight += v[th_feature]
            history_weight += v[cap_feature]
            history_weight += v[len_feature]

            for j in range(1,4):
                if len(words[i-1]) >= j:
                    suffix_feature = ":".join(["SUFF", words[i-1][-j:], str(j), t1])
                    history_weight += v[suffix_feature]
                else:
                    break

            scored_histories += history.strip() + " " + str(history_weight) + "\n"
    
        # compute highest scoring tag sequence
        bigram_tag_sequence = call(decoder_server, scored_histories.strip())
        bigram_tags = bigram_tag_sequence.strip().split("\n")

        # format tagging output
        tagged_output = ""
        bigram_tags.pop() # remove STOP tag
        for bigram_tag in bigram_tags:
            i, t0, t1 = bigram_tag.strip().split(" ")
            tagged_output += "\t".join([words[int(i)-1], t1]) + "\n" 

        print(tagged_output)

    dev_file.close()
