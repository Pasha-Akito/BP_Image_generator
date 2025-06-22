import pandas as pd
import csv

def extract_folder_name(bp_number):
    return f"p{bp_number:03d}"

def get_image_pair_identifers():
    return [(left, right) for right in range(6, 12) for left in range(6)]

def get_image_pair_identifers_for_only_three_images():
    return [(left, right) for right in range(3, 6) for left in range(3)]

def generate_data():
    simple_sentence_image_relationships_df = pd.read_csv("../data/simple_sentence_image_relationships.csv")
    expanded_sentence_image_relationships = []
    image_pair_identifiers = get_image_pair_identifers()
    for row in simple_sentence_image_relationships_df.itertuples():
        sentence = row.sentence
        bp_number = row.bp_number
        folder_name = extract_folder_name(bp_number)

        if (bp_number > 999): 
            folder_name = f"p{bp_number}"
        
        if (bp_number == 1211 or bp_number == 1232): # 1211 & 1232 only have three images each
            image_pair_identifiers = get_image_pair_identifers_for_only_three_images()

        for left_identifer, right_identifier in image_pair_identifiers:
            expanded_sentence_image_relationships.append({
                "sentence": f"{sentence}",
                "left_image": f"../bp_images/{folder_name}/{left_identifer}.png",
                "right_image": f"../bp_images/{folder_name}/{right_identifier}.png"
            })

    expanded_sentence_image_relationships_df = pd.DataFrame(expanded_sentence_image_relationships)
    expanded_sentence_image_relationships_df.to_csv("../data/expanded_sentence_image_relationships.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    generate_data()