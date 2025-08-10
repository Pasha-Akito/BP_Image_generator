import pandas as pd
import csv
from config import DATASET

def extract_folder_name(bp_number):
    return f"p{bp_number:04d}"

def get_image_pair_identifers():
    return [(left, right) for right in range(6, 12) for left in range(6)]

def generate_data():
    print("Using dataset:", DATASET)
    simple_sentence_image_relationships_df = pd.read_csv(f"../data/{DATASET}_words_data/{DATASET}_words_image_relationships.csv")
    expanded_sentence_image_relationships = []
    image_pair_identifiers = get_image_pair_identifers()
    for row in simple_sentence_image_relationships_df.itertuples():
        sentence = row.sentence
        bp_number = row.bp_number
        folder_name = extract_folder_name(bp_number)

        for left_identifer, right_identifier in image_pair_identifiers:
            expanded_sentence_image_relationships.append({
                "sentence": f"{sentence}",
                "left_image": f"../bp_images/{folder_name}/{left_identifer}.png",
                "right_image": f"../bp_images/{folder_name}/{right_identifier}.png"
            })

    expanded_sentence_image_relationships_df = pd.DataFrame(expanded_sentence_image_relationships)
    expanded_sentence_image_relationships_df.to_csv(F"../data/{DATASET}_words_data/expanded_{DATASET}_words_image_relationships.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    generate_data()