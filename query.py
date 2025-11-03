import os
import pandas as pd
import numpy as np
import classla
import string
from collections import Counter
import json

# Initialize the CLASSLA NLP pipeline for Croatian
nlp = classla.Pipeline(lang='hr', processors='tokenize, pos, lemma')

# Load stopwords
with open('stopwords-hr.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip().lower() for line in f if line.strip())

def clean_lemmatize(text):
    '''Preprocess text: lemmatize, lowercase, remove punctuation and stopwords'''
    doc = nlp(text)
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma.lower().strip()
            if (
                lemma and
                lemma not in string.punctuation and
                lemma not in stopwords
            ):
                lemmas.append(lemma)
    return lemmas

def create_td_matrix(data, vocab):
    '''Create term-document frequency matrix'''
    td_matrix = pd.DataFrame(0, index=vocab, columns=data['id'])
    for doc_id, lemmas in zip(data['id'], data['body']):
        counts = Counter(lemmas)
        for lemma, freq in counts.items():
            if lemma in td_matrix.index:
                td_matrix.at[lemma, doc_id] = freq
    return td_matrix

def bm25_score(term_doc_matrix, query_lemmas, idf_series, document_index, k1=1.5, b=0.75):
    '''Calculate BM25 scores for a given query'''
    doc_lengths = term_doc_matrix[document_index].sum(axis=0)
    avgdl = doc_lengths.mean()
    scores = pd.Series(0.0, index=document_index)
    for term in set(query_lemmas):
        if term not in term_doc_matrix.index:
            continue
        tf = term_doc_matrix.loc[term, document_index]
        idf = idf_series.get(term, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_lengths / avgdl)
        scores += idf * numerator / (denominator + 1e-10)
    return scores.sort_values(ascending=False)

def retrieve_index(data, scores, document_index):
    '''Retrieve top 5 document IDs by BM25 score'''
    data = data.set_index(document_index)
    data['scores'] = scores
    return data.sort_values('scores', ascending=False).head(5).index.tolist()

def display_document(doc_id):
    with open("individual_data.json", "r", encoding="utf-8") as f:
        original_data = json.load(f)
    for item in original_data:
        if item["id"] == doc_id:
            print(f"\n=== {item['title']} ===\n")
            print(item['body'])
            return
    print("No document found with that ID.")

def main():
    # Load or preprocess data
    prepped_file = 'prepped_data.pkl'
    if os.path.exists(prepped_file):
        print("Loading cached preprocessed data...")
        df = pd.read_pickle(prepped_file)
    else:
        print("Preprocessing data from scratch (can take ~20 mins)...")
        df = pd.read_json('individual_data.json')
        df = df.drop('title', axis=1)
        df['body'] = df['body'].apply(clean_lemmatize)
        df.to_pickle(prepped_file)

    vocab = sorted(set(word for body in df['body'] for word in body))
    term_doc_matrix = create_td_matrix(df, vocab)

    # Compute IDF
    document_index = df['id']
    doc_freq = (term_doc_matrix[document_index] > 0).sum(axis=1)
    idf_series = np.log2(len(document_index) / doc_freq.replace(0, 1))

    # main query loop
    while True:
        query = input('\nType in a query (type exit() to quit): ')
        if query.lower() == "exit()":
            print("Exiting program")
            break

        qlemmas = clean_lemmatize(query)
        bm25_scores = bm25_score(term_doc_matrix, qlemmas, idf_series, document_index)
        top_doc_ids = retrieve_index(df, bm25_scores, 'id')

        # Display top results
        print("\nTop 5 documents:")
        for doc_id in top_doc_ids:
            print(doc_id)

        see_content = input("\nEnter a document ID to view the content, or type exit() to quit: ")
        if see_content.lower() == "exit()":
            print("Exiting program")
            break

        selected_id = int(see_content)
        display_document(selected_id)

if __name__ == "__main__":
    main()