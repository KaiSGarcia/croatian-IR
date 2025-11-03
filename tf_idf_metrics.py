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
    '''Creates a term-document frequency matrix'''
    td_matrix = pd.DataFrame(0, index=vocab, columns=data['id'])
    for doc_id, lemmas in zip(data['id'], data['body']):
        counts = Counter(lemmas)
        for lemma, freq in counts.items():
            if lemma in td_matrix.index:
                td_matrix.at[lemma, doc_id] = freq
    return td_matrix

def create_tfidf_matrix(vocab_index, document_index, idf_series):
    tf = np.log2(1 + vocab_index[document_index])
    return tf.mul(idf_series, axis=0)

def query_vector(query_lemmas, vocab_index, idf_series):
    query_counts = Counter(query_lemmas)
    tf_query = pd.Series(0.0, index=vocab_index.index)
    for term, freq in query_counts.items():
        if term in idf_series.index:
            tf_query[term] = np.log2(1 + freq) * idf_series[term]
    return tf_query

def cosine_similarity(tfidf_matrix, query_vector):
    dot_products = tfidf_matrix.T.dot(query_vector)
    query_norm = np.sqrt((query_vector ** 2).sum())
    doc_norms = np.sqrt((tfidf_matrix ** 2).sum(axis=0))
    return (dot_products / (query_norm * doc_norms + 1e-10)).sort_values(ascending=False)

def retrieve_index(data, cosine_scores, document_index):
    data = data.set_index(document_index)
    data['scores'] = cosine_scores
    return data.sort_values('scores', ascending=False).head(5).index.tolist()


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

    # Compute IDF and TF-IDF
    document_index = df['id']
    doc_freq = (term_doc_matrix[document_index] > 0).sum(axis=1)
    idf_series = np.log2(len(document_index) / doc_freq.replace(0, 1))
    tfidf_matrix = create_tfidf_matrix(term_doc_matrix, document_index, idf_series)

    x = pd.read_json('individual_data.json')
    test_dict = dict(zip(x['id'], x['title'])) # Build dictionary from 'id' to 'title'

    counter = 0
    total = len(test_dict)
    ranks = []
    average_precisions = []

    fours_fives = pd.read_json('high_similarity_data.json')

    total_fourfive = 0
    fourfive_count = 0
    
    # use titles as queries to measure accuracy
    for doc_id, title in test_dict.items():
        qlemmas = clean_lemmatize(title)
        qvec = query_vector(qlemmas, term_doc_matrix, idf_series)
        similarities = cosine_similarity(tfidf_matrix, qvec)
        top_doc_ids = retrieve_index(df, similarities, 'id') # get top 5 doc IDs, ranked

        # Accuracy@5 and Rank tracking
        if doc_id in top_doc_ids:
            counter += 1
            rank = top_doc_ids.index(doc_id) + 1
            ranks.append(rank)
            average_precisions.append(1 / rank)  # AP for this query
        else:
            average_precisions.append(0.0)

        # Calculate similarity score usefulness
        for doc_id_candidate in top_doc_ids:
            for _, d in fours_fives.iterrows():
                total_fourfive += 1
                if doc_id_candidate == d["id"] and d["id2"] not in top_doc_ids: # if paired doc not in top 5 returned
                    fourfive_count += 1

    # Final metrics
    accuracy = counter / total
    avg_rank = sum(ranks) / len(ranks) if ranks else None
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    usefulness = fourfive_count / total_fourfive if total_fourfive > 0 else 0.0

    # Output
    print(f"Found correct doc in top 5 for {counter}/{total} queries.")
    print(f"Accuracy@5: {accuracy:.2%}")
    if avg_rank is not None:
        print(f"Average rank position (for successful hits): {avg_rank:.2f}")
    else:
        print("No correct documents found in top 5; cannot compute average rank.")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Similarity score usefulness: {usefulness:.4f}")

if __name__ == "__main__":
    main()