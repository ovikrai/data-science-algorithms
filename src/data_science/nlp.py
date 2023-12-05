from sklearn.feature_extraction.text import CountVectorizer
import nltk

# NATURAL LANGUAGE PROCESSING TOOLS #############################
# Creating the Bag of Words Model (Vectorization)
# TODO: TRANSFORM TO CLASS
TextCorpus = list[str]
EMPTY_STRING = ''
SPACE = ' '
RETURN = '\n'


def load_corpus(file_path: str) -> str:
    with open(file_path, 'r') as file:
        raw_corpus = file.read()

    return raw_corpus


# TOKENIZE: DIVIDE CORPUS TEXT IN WORDS
def tokenize(raw_corpus: str, lang='english') -> list:
    return nltk.tokenize.word_tokenize(raw_corpus, language=lang)


def replace_symbol(corpus: TextCorpus, key_symbol: chr, new_symbol: chr):
    n = len(corpus)

    for i in range(0, n):
        corpus[i].replace(key_symbol, new_symbol)

    return corpus


def replace_token(corpus: TextCorpus, key_token: str, new_token: str):
    n = len(corpus)

    for i in range(0, n):
        item_token = corpus[i]
        if item_token is key_token:
            corpus[i] = new_token

    return corpus


def remove_symbol(corpus: TextCorpus, key_symbol: chr):
    n = len(corpus)

    for i in range(0, n):
        corpus[i].replace(key_symbol, EMPTY_STRING)

    return corpus


def remove_token(corpus: TextCorpus, key_token: str):
    n = len(corpus)

    for i in range(0, n):
        if corpus[i] is key_token:
            corpus.remove(key_token)

    return corpus


def vectorize(x_train, max_features=1500):
    cv = CountVectorizer(max_features=max_features)
    return cv.fit_transform(x_train).toarray()
