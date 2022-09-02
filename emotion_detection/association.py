import pandas as pd
import re
from collections import defaultdict, OrderedDict

EMO_LIST = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise"]


def load_data(nrc_path, data_path):
    emolex_df = pd.read_csv(nrc_path, names=["word", "emotion", "association"], skiprows=1, sep='\t')
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    data = pd.read_csv(data_path)
    return emolex_words, data


def get_emotion(emotion, emolex_words, data):
    emo_dict = emolex_words.set_index('word').to_dict()[emotion]
    emo = data['utterance'].apply(lambda x: sum([emo_dict[i] if i in emo_dict else 0 for i in x]))
    return emo


def get_emo_words(data, emolex_words):
    emo_dfs = {}
    for emo in EMO_LIST:
        emo_dfs[emo] = get_emotion(emo, emolex_words, data)
    lex = pd.DataFrame(
        {'utterance': data['utterance'], 'anger': emo_dfs['anger'], 'anticipation': emo_dfs['anticipation'],
         'disgust': emo_dfs['disgust'], 'fear': emo_dfs['fear'],
         'joy': emo_dfs['joy'], 'sadness': emo_dfs['sadness'],
         'surprise': emo_dfs['surprise']})
    lex['emo_lex'] = ''
    lex['max_lex'] = lex.loc[:, lex.columns != 'utterance'].max(axis=1)
    data_lex = data.join(lex.loc[:, lex.columns!='utterance'])
    data_lex = data_lex[data_lex["max_lex"] > 0]
    for emo in EMO_LIST:
        data_lex['emo_lex'] = data_lex.apply(lambda row: str(row['emo_lex'] + " " + emo)
        if row[emo] == row['max_lex']
        else str(row['emo_lex']),
                                             axis=1)
    data_lex['emo_lex'] = data_lex['emo_lex'].apply(lambda x: ','.join(x.lstrip().split(' ')))
    return (data_lex)


def get_emo_distribution(data):
    orig_dic = defaultdict(int)
    lex_dic = defaultdict(int)
    for l in data['emo8']:
        labels = l.split(',')
        for i in labels:
            orig_dic[i] += 1
    for l in data['emo_lex']:
        labels = l.split(',')
        for i in labels:
            lex_dic[i] += 1
    orig_dic = OrderedDict(sorted(orig_dic.items()))
    lex_dic = OrderedDict(sorted(lex_dic.items()))
    return (orig_dic, lex_dic)


def main():
    lexicon, data = load_data("NRC-Emotion.txt", path_to_data)
    data['utterance'] = data['utterance'].apply(lambda x: re.split("\W+", str(x).lower())) # prepare and tokenize
    data_lex = get_emo_words(data, lexicon)
    print(data_lex)
    orig_d, lex_d = get_emo_distribution(data_lex)

if __name__ == "__main__":
    main()