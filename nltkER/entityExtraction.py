'''
Demonstration for named entity extraction learning with limited supervised data

Data for demo from Kaggle IMDB movie review dataset

Author:  Daniel Marasco


TODO:
-make current_chunk check and append into function
-Set up cleaning
-Set up ensemble and different methods (districtdatalabs tutorial)
'''

import pandas as pd
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
from nltk.tree import Tree


def get_paragraph_chunks(text):
    sents = sent_tokenize(text)
    unmerged = [get_continuous_chunks(sent) for sent in sents]
    return list(set([val for sublist in unmerged for val in sublist]))


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join(
                                    [token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                    continue
    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
            current_chunk = []

    return continuous_chunk


def main():
    examples = 10
    text = 'review'

    train = pd.read_csv("../data/labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)
    # test = pd.read_csv("../data/testData.tsv", header=0, delimiter="\t",
    #                    quoting=3)
    # unlabeled_train = pd.read_csv("../data/unlabeledTrainData.tsv", header=0,
    #                               delimiter="\t", quoting=3)

    data = train.loc[0:examples, :].copy()
    data['entities'] = data[text].map(get_paragraph_chunks)
    print(data.loc[:5, 'entities'])
    return data

if __name__ == "__main__":
    entities = main()
