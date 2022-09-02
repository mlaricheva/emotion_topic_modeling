import pandas as pd
import re
import string
import json
from nltk.corpus import stopwords
import nltk
import gensim
from gensim.utils import simple_preprocess
import numpy as np
import warnings
from wordcloud import WordCloud

CUSTOM_STOP_WORDS = ["like", "really", "think", "mhm", "kind", "mean"]
EMO_LIST = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise"]


def load_data(nrc_path, data_path):
    emolex_df = pd.read_csv(nrc_path,  names=["word", "emotion", "association"], skiprows=1, sep='\t')
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    data = pd.read_csv(data_path)
    return emolex_words, data


def remove_stop_words(utterance, stop_words):
    utterance=gensim.utils.simple_preprocess(utterance)
    return ' '.join([word for word in utterance if word not in stop_words])


def clean_and_tokenize(text, stop_words):
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    text = text.lower() # lower the text
    with open('contractions.json') as f: # replace contractions
        contractions = json.load(f)
    for word in text.split():
        if word in contractions:
            text = text.replace(word,contractions[word])
    text = remove_stop_words(text, stop_words)
    text=re.sub('<.*?>', ' ', text)   # remove html tags
    text = text.translate(str.maketrans(' ',' ',string.punctuation))
    text = re.sub('[^a-zA-Z]',' ',text)  # add space where required
    tokens = text.split()
    pos = nltk.pos_tag(tokens) # clean using pos-tagger. see https://www.guru99.com/pos-tagging-chunking-nltk.html
    for p in pos:
        if p[1] in ["CC", "CD", "DT", "EX", "IN", "LS", "MD", "NNP", "PDT", "POS", "PRP", "PRP$", "RP", "TO",
                    "WDT", "WP", "WRB"]:
            tokens.remove(p[0])
    return tokens


def data_preparation(data):
    stop_words = stopwords.words('english')
    stop_words.extend(CUSTOM_STOP_WORDS)
    data['utterance'] = data['utterance'].apply(lambda x: clean_and_tokenize(x, stop_words))
    return(data)

def get_emo_dict(lexicon, emo):
    emo_dict = lexicon.set_index('word').to_dict()[emo]
    emo_dict = {k: emo_dict[k] for k in emo_dict if not np.isnan(emo_dict[k])}
    return(emo_dict)

def get_emo_words(data, lexicon):
    original_cols = data.columns.copy()
    for emo in EMO_LIST:
        emo_dict = get_emo_dict(lexicon, emo)
        data[emo] = data["utterance"].apply(lambda x: sum([emo_dict[i] if i in emo_dict else 0 for i in x]))
        data[emo + "_words"] = data["utterance"].apply(lambda x: ' '.join([i for i in x if i in emo_dict]))
    ### finding the most intense emotions
    data['max_lex'] = data.loc[:, ~data.columns.isin(original_cols)].max(axis=1)
    data = data[data["max_lex"] > 0]
    data['emo_lex'] = ''
    for emo in EMO_LIST:
        data['emo_lex'] = data.apply(
            lambda row: str(row['emo_lex'] + " " + emo) if row[emo] == row['max_lex'] else str(row['emo_lex']), axis=1)
    return data


def get_words(data, lexicon, emo, threshold=1, cutoff = 0.3, sort='association'):
    emo_words = data[emo + '_words'].copy()
    emo_words = emo_words[emo_words != '']  # leaving only not empty entries
    emo_words = emo_words.str.split(expand=True).stack()
    emo_df = pd.DataFrame({'word': emo_words})  # creating dataframe
    emo_dict = get_emo_dict(lexicon, emo)
    emo_df['association'] = emo_df['word'].apply(lambda x: emo_dict[x])  # adding association
    emo_df = emo_df[emo_df['association'] >= cutoff]  # cut-off of 0.3 for association
    emo_df.reset_index(drop=True, inplace=True)
    emo_df['counts'] = emo_df.groupby(['word'])['association'].transform('count')  # adding counts
    emo_df = emo_df[emo_df['counts'] >= threshold]  # cut-off of 1 for counts
    emo_df = emo_df[['word', 'association', 'counts']].drop_duplicates(subset=['word'])  # drop dublicates
    return (emo_df.sort_values(by=[sort], ascending=False))


def common_words(data, threshold=1, cutoff = 0.3, sort='counts'):
    emo_dfs=[]
    for emo in EMO_LIST:
        df = get_words(data, emo, threshold, cutoff, sort)
        df["emo"] = emo
        emo_dfs.append(df)
    words_df = pd.concat(emo_dfs)
    return(words_df.sort_values(by=['counts', 'association'], ascending=False))

def orig_distr(subset):
    emo8_list = [i.split(',') for i in subset["emo8"].tolist()]
    emo8 = [i for j in emo8_list for i in j]
    emo8 = pd.Series(emo8)
    return(emo8.value_counts(normalize=True).round(3))

def lex_distr(subset):
    emo_list = [i.replace(' ', ',').split(',') for i in subset["emo_lex"].tolist()]
    emo = [i for j in emo_list for i in j]
    emo = pd.Series(emo)
    return(emo.value_counts(normalize=True).round(3))

def avg_intensity(subset):
    intensities = []
    for emo in EMO_LIST:
        intensities.append(round(subset[emo].mean(), 3))
    return pd.DataFrame(EMO_LIST, intensities)


def main():
    warnings.filterwarnings("ignore")
    lexicon, data = load_data("NRC-Emotion-Intensity.txt", path_to_data)
    data = data_preparation(data)
    data_lex = get_emo_words(data, lexicon)

    # PEERS: Time 1 VS Time 3
    peers_subset = data_lex[data_lex["transcript"]<13000]
    time1_peers = peers_subset[peers_subset["transcript"] % 10 == 1] # time1
    time3_peers = peers_subset[peers_subset["transcript"] % 10 == 3] # time3
    get_words(peers_subset, lexicon, 'anticipation').head(10) # get 10 most common words for anticipation
    orig_distr(peers_subset) # get the distribution of original labels
    lex_distr(peers_subset) # get the distribution of the labels based on lexicon scores
    avg_intensity(peers_subset) # get the average emotion intensity of the subset



if __name__ == "__main__":
    main()