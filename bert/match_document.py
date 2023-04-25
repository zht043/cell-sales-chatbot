import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_document():
    with open("dataset/train.json", 'r') as f:
        data = json.load(f)
    documment = []
    paragraphs = data['data'][0]['paragraphs']
    for (index, paragraph) in enumerate(paragraphs):
        context = paragraph['context']
        documment.append(context)

    return documment


def match_document(question):
    question = question
    document = get_document()
    document.insert(0, question)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(document)
    cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
    cosine_sim_question = cosine_sim[0]
    arr = np.array(cosine_sim_question)
    document_index = np.argsort(-arr)[1]
    context = document[document_index]

    return context





if __name__ == "__main__":
    match_document()