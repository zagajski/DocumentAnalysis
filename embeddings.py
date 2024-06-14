import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument

def glove(corpus):
    print("started gloVe")
    glove_vectors = api.load("glove-wiki-gigaword-100")
    vectors = []
    for doc in corpus:
        word_vectors = [glove_vectors[word] for word in doc.split() if word in glove_vectors]
        if word_vectors:  # Check if word_vectors is not empty
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(glove_vectors.vector_size))  # Append a zero vector if no words are found
    return np.array(vectors), 'GloVe', None

def word2vec(corpus):
    print("started word2Vec")
    corpus_splitted = [doc.split() for doc in corpus]
    model = Word2Vec(sentences=corpus_splitted, vector_size=100, window=5, min_count=1, epochs=10)
    
    vectors = []
    for doc in corpus:
        word_vectors = [model.wv[word] for word in doc.split() if word in model.wv]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))  # Append a zero vector if no words are found
    
    return np.array(vectors), 'Word2Vec', model

def tfidf(corpus):
    print("started tfidf")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), 'TF-IDF', vectorizer

def doc2vec(corpus):
    print("started doc2vec")
    tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(vector_size=20, min_count=1, epochs=20)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = [model.infer_vector(doc.split()) for doc in corpus]
    return np.array(vectors), 'Doc2Vec', model

def embeddingWith(index, corpus, return_vectorizer):
    if index == 0:
        vector, embedding_type, vectorizer = glove(corpus)
    elif index == 1:
        vector, embedding_type, vectorizer = word2vec(corpus)
    elif index == 2:
        vector, embedding_type, vectorizer = tfidf(corpus)
    elif index == 3:
        vector, embedding_type, vectorizer = doc2vec(corpus)
    else:
        raise ValueError("Invalid index. Choose a number between 0 and 3.")
    
    if not return_vectorizer:
        return vector, embedding_type
    else:
        return vector, embedding_type, vectorizer
