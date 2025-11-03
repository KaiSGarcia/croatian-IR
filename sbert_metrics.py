import os
import pandas as pd
import numpy as np
import classla
import string
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import torch

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

def bm25_score(term_doc_matrix, query_lemmas, idf_series, document_ids, doc_lengths, k1=1.5, b=0.75):
    scores = pd.Series(0.0, index=document_ids)
    avgdl = doc_lengths.mean()
    for term in set(query_lemmas):
        if term not in term_doc_matrix.index:
            continue
        tf = term_doc_matrix.loc[term, document_ids]
        idf = idf_series.get(term, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_lengths / avgdl)
        score = idf * numerator / (denominator + 1e-10)
        scores += score
    return scores.sort_values(ascending=False)

def bm25_retrieval(query_lemmas, term_doc_matrix, idf_series, document_ids, doc_lengths, top_k=5):
    scores = bm25_score(term_doc_matrix, query_lemmas, idf_series, document_ids, doc_lengths)
    top_doc_ids = scores.head(top_k).index.tolist()
    return top_doc_ids, scores.loc[top_doc_ids].values

def sbert_rerank(query, doc_ids, original_texts, sbert_model, top_k=5):
    doc_texts = [original_texts.get(doc_id, "") for doc_id in doc_ids]

    valid_doc_ids = []
    valid_texts = []
    for doc_id, text in zip(doc_ids, doc_texts):
        if isinstance(text, str) and text.strip():
            valid_doc_ids.append(doc_id)
            valid_texts.append(text)

    if not valid_texts:
        return []

    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = sbert_model.encode(valid_texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(valid_doc_ids)))
    return [valid_doc_ids[i] for i in top_results.indices.tolist()]


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
    document_ids = df['id'].values
    doc_lengths = term_doc_matrix[document_ids].sum(axis=0)

    # Compute IDF
    doc_freq = (term_doc_matrix[document_ids] > 0).sum(axis=1)
    idf_series = np.log2(len(document_ids) / doc_freq.replace(0, 1))
    
    # SBERT reranking
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    original_texts = pd.read_json('individual_data.json').set_index('id')['body'].to_dict()

    x = pd.read_json('individual_data.json')
    test_dict = dict(zip(x['id'], x['title']))

    counter = 0
    total = len(test_dict)
    ranks = []
    average_precisions = []

    fours_fives = pd.read_json('high_similarity_data.json')
    total_fourfive = 0
    fourfive_count = 0

    for doc_id, title in test_dict.items():
        qlemmas = clean_lemmatize(title)
        top_doc_ids, _ = bm25_retrieval(qlemmas, term_doc_matrix, idf_series, document_ids, doc_lengths, top_k=5) # change to control # of reranked docs (n.b. higher = worse accuracy)
        reranked_doc_ids = sbert_rerank(title, top_doc_ids, original_texts, sbert_model, top_k=5)

        # Accuracy@5 and Rank tracking
        if doc_id in reranked_doc_ids:
            counter += 1
            rank = reranked_doc_ids.index(doc_id) + 1
            ranks.append(rank)
            average_precisions.append(1 / rank)  # AP for this query
        else:
            average_precisions.append(0.0)

        # Usefulness check for 4s and 5s
        for doc_id_candidate in reranked_doc_ids:
            for _, d in fours_fives.iterrows():
                total_fourfive += 1
                if doc_id_candidate == d["id"] and d["id2"] not in reranked_doc_ids:
                    fourfive_count += 1

    # Results
    accuracy = counter / total
    avg_rank = sum(ranks) / len(ranks) if ranks else None
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    usefulness = fourfive_count / total_fourfive if total_fourfive > 0 else 0.0

    print(f"\nFound correct doc in top 5 for {counter}/{total} queries.")
    print(f"Accuracy@5: {accuracy:.2%}")
    if avg_rank is not None:
        print(f"Average rank position (for successful hits): {avg_rank:.2f}")
    else:
        print("No correct documents found in top 5; cannot compute average rank.")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Similarity score usefulness: {usefulness:.4f}")

if __name__ == "__main__":
    main()

