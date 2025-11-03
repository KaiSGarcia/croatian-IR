import os
import json

input_folder = "master_data"
all_data = []

# Loop over each file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file) # that file's list of dict objects
            for obj in data:
                all_data.append(obj) # add each object to master list

def remove_unneeded_keys(dict_list):
    '''Removes all keys from JSON objects except IDs, titles, and bodies'''
    unneeded_keys = ["portal", "date_published", "portal2", "date_published2", "annotator", "annotation_id", "created_at", "updated_at", "lead_time"]
    for d in dict_list:
        for key in unneeded_keys:
            if key in d:
                del d[key]
    return dict_list

clean_data = remove_unneeded_keys(all_data)

# Pairs of articles with similarity score 4 or 5
fours_fives = []
for obj in clean_data:
    if (str(obj.get("choice")) == "5") or (str(obj.get("choice")) == "4"):
        fours_fives.append(obj)

# Single articles
def create_individual_list(dict_list):
    '''Creates a list of all individual articles from original paired article JSON objects''' 
    indiv_list = []
    seen_titles = set()

    # remove similarity scores for single articles
    for d in dict_list:
        key = "choice" 
        if key in d:
            del d[key]

        # first article (id, title, body)
        article1 = {
            "id": d["id"],
            "title": d["title"],
            "body": d["body"]
        }

        # second article (id2, title2, body2 renamed to id, title, body) 
        article2 = {
            "id": d["id2"],
            "title": d["title2"],
            "body": d["body2"]
        }

        # prevent duplicates when making list of all single articles
        if article1["title"] not in seen_titles:
            indiv_list.append(article1)
            seen_titles.add(article1["title"])

        if article2["title"] not in seen_titles:
            indiv_list.append(article2)
            seen_titles.add(article2["title"])

    return indiv_list

individual_data = create_individual_list(clean_data)

for i, article in enumerate(individual_data):
    article['id'] = i + 1 # Assign new ids starting from 1

def map_article_ids(individual_data, fours_fives):
    '''Maps new IDs assigned to single articles to the same articles in the high similarity dataset to keep IDs consistent'''
    for article in individual_data:
        for article_pair in fours_fives:
            if article_pair["title"] == article["title"]:
                article_pair["id"] = article["id"]
            if article_pair["title2"] == article["title"]:
                article_pair["id2"] = article["id"]
    return fours_fives

updated_fours_fives = map_article_ids(individual_data, fours_fives)

with open("high_similarity_data.json", "w", encoding="utf-8") as output_file:
    json.dump(updated_fours_fives, output_file, ensure_ascii=False, indent=2)

with open("individual_data.json", "w", encoding="utf-8") as output:
    json.dump(individual_data, output, ensure_ascii=False, indent=2)