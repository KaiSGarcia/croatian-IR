# croatian-IR
Croatian Information Retrieval Systems

This project compares performance of three information retrieval systems for a Croatian news article corpus. 
 - TF-IDF
 - BM25
 - Dual stage: BM25 for retrieval, SBERT for re-ranking

## How to use
- First run extract_data.py to generate individual_data.json and high_similarity.json
- The metrics scripts run the specified system using article titles as queries, returning accuracy and performance stats.
- query.py works like a search engine. It is built on the highest performing system (BM25). Users enter their query and it returns the top 5 relevant articles, with an option to view article content.
- Running any system for the first time will take at least 20 mins. The scripts cache the preprocessed data to prepped_data.pkl, which makes running (or re-running) the remaining scripts much faster.

## Data
This project utilizes a subset of the data (found at /sampled_data/annotations/) from [Article-BERTi´c](https://github.com/ir2718/article-bertic). It consists of 2,109 recent Croatian news articles from various websites. 

### What are "similarity scores"?

Each JSON element in the original annotated data contains two news articles and a similarity score metric called "choice" (ranging from 0-5) that measures how closely related the articles are. 5 means that the articles are completely equivalent and express the same meaning, while 0 means the articles are completely unrelated. 

This was the inspiration for a custom value in the metrics scripts called "similarity score usefulness." The initial data extraction (extract_data.py) creates a list of the individual articles that have "choice" of 4 or 5 (high_similarity.json). When the metrics script retrieves a document, it checks if it is present in high_similarity.json. If it is, it then checks if the article it was originally paired with, appears in the IR system's five retrieved results. Finally, it calculates the percentage of times the paired article did NOT appear. A higher score implies the model frequently misses the second document when retrieving the first, and thus having similarity scores between article pairs would be a useful feature for IR systems.
