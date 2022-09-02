from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
import json
import re
import string
import nltk
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import warnings
import matplotlib.pyplot as plt
import spacy
import math

CUSTOM_STOP_WORDS = ["oh", "uh", "yeah", "well", "um", "umm", "like", "really", "think", "mhm", "mm",
                       "know", "right", "go", "get", "got", "going", "cause", "okay", "ok", "thing", "stuff", "one",
                       "kind", "mean",
                       "want", "time", "hmm", "mhmm", "thing", "something", "things", "say", "said", "talk", "lot",
                       "even", "still",
                       "would", "good", "people", "see", "much", "take", "make", "need", "day", "cos", "yes", "gon"]

DOCUMENT_SIZE = 10
N_TOP_WORDS = 10
N_TOPICS = 5

def data_prepare(data, n):
    data["grp_idx"] = range(len(data))
    data["grp_idx"] = data["grp_idx"] // n
    data_df = data.groupby(['grp_idx'])['utterance'].apply(' '.join).reset_index()
    return (data_df)

def remove_punct(df):
    df['utterance']=df['utterance'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
    df['utterance']=df['utterance'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
    df['utterance']=df['utterance'].apply(lambda x: re.sub('[,\.!?]', '', x))
    df['utterance']=df['utterance'].apply(lambda x: x.replace("...", ""))
    df['utterance']=df['utterance'].apply(lambda x: x.lower())
    return df

def text_cleaning(text):
    with open('../contractions.json') as f: # replace contractions
        contractions = json.load(f)
    for word in text.split():
        if word in contractions:
            text = text.replace(word,contractions[word])
    return text

def lemmatize(utterance, spacy_en):
    lemma = spacy_en(utterance)
    return(" ".join([token.lemma_ for token in lemma if token.lemma_ != "-PRON-"]))

def remove_stop_words(utterance, stop_words):
    utterance=gensim.utils.simple_preprocess(utterance)
    return ' '.join([word for word in utterance if word not in stop_words])

def clean_and_lemmatize(df):
    df = remove_punct(df)
    df['utterance']=df['utterance'].apply(lambda x: text_cleaning(x))
    df['pos_tags']=df['utterance'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
    df['utterance']=df['pos_tags'].apply(lambda x: [i[0] for i in x if i[1] not in
                                                    ["CC", "CD", "DT", "EX", "IN", "LS", "MD", "NNP", "PDT", "POS",
                                                     "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WRB",
                                                     ",", ".", '()', "RB", "UH"]]) # clean using pos-taggers
    df['utterance'] = df['utterance'].apply(lambda x: ' '.join(x))
    spacy_en = spacy.load('en_core_web_sm') # lemmatize
    df['utterance'] = df['utterance'].apply(lambda x: lemmatize(x, spacy_en))
    stop_words = stopwords.words('english') # remove stop words
    stop_words.extend(CUSTOM_STOP_WORDS)
    df['utterance'] = df['utterance'].apply(lambda x: remove_stop_words(x, stop_words))
    return df


def make_wordcloud(utterance):
    all_utt = ','.join(list(utterance.values))
    wordcloud = WordCloud(width=800, height=400, max_words=5000, contour_width=3,background_color='white')
    wordcloud.generate(all_utt)
    wordcloud.to_file('wordcloud.png')
    return wordcloud

def return_next(x):
    return(math.ceil(x/2.)*2)

def plot_words(model, feature_names, n_top_words, n_topics, topics):
    n_facets = return_next(n_topics)
    fig, axes = plt.subplots(2, int(n_facets/2), figsize=(25, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(topics[topic_idx],
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=25)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        #fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.85, bottom=0.05, wspace=0.4, hspace=0.2)
    return fig


def get_top_words(nmf, tfidf):
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [tfidf[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        return(top_features)

def main():
    # reading and cleaning data
    raw_data = pd.read_csv(path_to_data)
    df = data_prepare(raw_data, DOCUMENT_SIZE)
    df = clean_and_lemmatize(df)

    # constructing a model
    data = df["utterance"].tolist()
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1,
                                       max_features=100,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data)
    nmf = NMF(n_components=N_TOPICS, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature = tfidf_vectorizer.get_feature_names()

    # results
    plot = plot_words(nmf, tfidf_feature, N_TOP_WORDS, N_TOPICS, topics = ["Topic " + str(i) for i in range(N_TOPICS)])
    plot.savefig("sample_plot.png", bbox_inches='tight')

    get_top_words(nmf, tfidf)

if __name__ == "__main__":
    main()