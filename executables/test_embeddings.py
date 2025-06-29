import torch
import torch.nn.functional as F
import clip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import chain

import sys
sys.path.append('../')

from model.loss_functions import CLIPLoss, PerceptualLoss, normalize_images, convert_to_grayscale, CLIP_MEAN, CLIP_STD
from data.dataset_loader import load_and_transform_image
from expand_data import extract_folder_name

IMAGE_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-B/32", device=device)
vgg = PerceptualLoss().to(device)


def get_image_features_from_vgg(vgg, image):
    image = convert_to_grayscale(image)
    image = normalize_images(image, vgg.VGG_MEAN, vgg.VGG_STD)
    image_features = []

    with torch.no_grad():
        for feature_extractor in vgg.feature_blocks:
            image = feature_extractor(image)
            image_features.append(image.flatten())

    return torch.cat(image_features)

def clip_similarity_two_images(image_path, image_2_path):
    image = load_and_transform_image(IMAGE_SIZE, image_path).to(device)
    image = image.unsqueeze(0)
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    image = convert_to_grayscale(image)
    image = normalize_images(image, CLIP_MEAN, CLIP_STD)

    image_2 = load_and_transform_image(IMAGE_SIZE, image_2_path).to(device)
    image_2 = image_2.unsqueeze(0)
    image_2 = F.interpolate(image_2, size=(224, 224), mode='bilinear', align_corners=False)
    image_2 = convert_to_grayscale(image_2)
    image_2 = normalize_images(image_2, CLIP_MEAN, CLIP_STD)
    
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(image)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        image_2_embeddings = clip_model.encode_image(image_2)
        image_2_embeddings = image_2_embeddings / image_2_embeddings.norm(dim=-1, keepdim=True)
        
        return ((image_embeddings @ image_2_embeddings.t())).item()

def get_clip_cosine_similarity_of_bongard_problem(bp_number):
    bp_folder_name = extract_folder_name(bp_number)
    image_paths_for_bp = [f"../bp_images/{bp_folder_name}/{i}.png" for i in range(12)]
    image_pair_matrix = [[img_path, image_path2] for img_path in image_paths_for_bp for image_path2 in image_paths_for_bp]
    flat_cosine_similarities = [clip_similarity_two_images(pair[0], pair[1]) for pair in image_pair_matrix]
    similarity_matrix = [flat_cosine_similarities[i * 12 : (i + 1) * 12] for i in range(12)]
    return similarity_matrix

def save_clip_cosine_similarity_of_first_100_bongard_problems():
    for i in range(100):
        similarity_matrix = get_clip_cosine_similarity_of_bongard_problem(i + 1)
        create_heatmap_of_cosine_matrix(similarity_matrix, i + 1, "CLIP")
        
def get_vgg_cosine_similarity(image_path, image_2_path):
    image = load_and_transform_image(IMAGE_SIZE, image_path).to(device)
    image_2 = load_and_transform_image(IMAGE_SIZE, image_2_path).to(device)
    image_features = get_image_features_from_vgg(vgg, image)
    image_2_feautures = get_image_features_from_vgg(vgg, image_2)
    image_features = image_features.unsqueeze(0)
    image_2_feautures = image_2_feautures.unsqueeze(0)
    return F.cosine_similarity(image_features, image_2_feautures).item()

def get_vgg_cosine_similarity_of_bongard_problem(bp_number):
    bp_folder_name = extract_folder_name(bp_number)
    image_paths_for_bp = [f"../bp_images/{bp_folder_name}/{i}.png" for i in range(12)]
    image_pair_matrix = [[img_path, image_path2] for img_path in image_paths_for_bp for image_path2 in image_paths_for_bp]
    flat_cosine_similarities = [get_vgg_cosine_similarity(pair[0], pair[1]) for pair in image_pair_matrix]
    similarity_matrix = [flat_cosine_similarities[i * 12 : (i + 1) * 12] for i in range(12)]
    return similarity_matrix


def create_heatmap_of_cosine_matrix(similarity_matrix, bp_number, title):
    plt.figure(figsize=(10, 8))
    image_labels = ["left1","left2","left3","left4","left5","left6","right1","right2","right3","right4","right5","right6"]
    percentage_annotation_matrix = [[f"{cosine_similarity * 100:.0f}" for cosine_similarity in row] for row in similarity_matrix]
    ax = sns.heatmap(similarity_matrix, xticklabels=image_labels, yticklabels=image_labels,annot=percentage_annotation_matrix,
        fmt="",square=True,cmap="inferno",vmin=0, vmax=1,cbar_kws={'ticks': [0, 0.5, 1]},annot_kws={'size': 10})
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['0%', '50%', '100%'])
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(f"{title} Similarity | Bongard Problem {bp_number}", fontsize=16, pad=20)
    ax.tick_params(axis='both',labelsize=10)
    ax.axvline(x=6, linewidth=1, color="white")
    ax.axhline(y=6, linewidth=1, color="white")
    plt.tight_layout()
    plt.savefig(f'../cosine_similarity/{title}/similarity_heatmap_bongard_problem_{bp_number}.png', dpi=300)
    print(f"Saved similarity_heatmap_bongard_problem_{bp_number}.png'")
    plt.close()

def save_vgg_cosine_similarity_of_first_100_bongard_problems():
    for i in range(100):
        similarity_matrix = get_vgg_cosine_similarity_of_bongard_problem(i + 1)
        create_heatmap_of_cosine_matrix(similarity_matrix, i + 1, "VGG")

def clip_similarity_between_image_and_text(image_path, text):
    image = load_and_transform_image(IMAGE_SIZE, image_path).to(device)
    image = image.unsqueeze(0)
    resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    gray_scale = convert_to_grayscale(resized)
    norm_image = normalize_images(gray_scale, CLIP_MEAN, CLIP_STD)
    
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(norm_image)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        text_embeddings = clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return ((image_embeddings @ text_embeddings.t())).item()

def save_text_image_clip_cosine_similarity_for_first_100_bongard_problems():
    similarity_matrix = []
    simple_dataset = pd.read_csv('../data/simple_sentence_image_relationships.csv')
    unique_sentences = simple_dataset['sentence'].unique()
    for i in range(100):
        for sentence in unique_sentences:
            similarity_matrix_of_bongard_problem = get_text_clip_cosine_similarity_of_bongard_problem(i + 1, sentence)
            similarity_matrix.append(similarity_matrix_of_bongard_problem)
        create_heatmap_of_clip_text(similarity_matrix, unique_sentences, i + 1)
        similarity_matrix = []
    return similarity_matrix

def get_text_clip_cosine_similarity_of_bongard_problem(bp_number, sentence):
    bp_folder_name = extract_folder_name(bp_number)
    image_paths_for_bp = [f"../bp_images/{bp_folder_name}/{i}.png" for i in range(12)]
    similarity_matrix = [clip_similarity_between_image_and_text(image, sentence) for image in image_paths_for_bp]
    return similarity_matrix

def create_heatmap_of_clip_text(cosine_similarities, unique_sentences, bp_number):
    fig, ax = plt.subplots(figsize=(6, 30))

    image_labels = ["left1","left2","left3","left4","left5","left6","right1","right2","right3","right4","right5","right6"]
    percentage_annotation_matrix = [[f"{cosine_similarity * 100:.0f}" for cosine_similarity in row] for row in cosine_similarities]

    sns.heatmap(
        cosine_similarities,
        ax=ax,
        square=True,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        cmap="cubehelix",
        xticklabels=image_labels,
        yticklabels=unique_sentences,
        annot=percentage_annotation_matrix,
        fmt="",
        annot_kws={"size": 4},
        cbar_kws={"shrink": 0.2}
        
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_title(f"CLIP Text Image Cosine Similarity | Bongard Problem {bp_number}", fontsize=16)
    ax.tick_params(axis='both',labelsize=5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f'../cosine_similarity/CLIP_TEXT_IMAGE/text_image_similarity_heatmap_bongard_problem_{bp_number}.png', dpi=300, bbox_inches='tight')
    print(f"Saved similarity_heatmap_bongard_problem_{bp_number}.png")
    plt.close()

if __name__ == "__main__":
    save_vgg_cosine_similarity_of_first_100_bongard_problems()
    save_clip_cosine_similarity_of_first_100_bongard_problems()
    save_text_image_clip_cosine_similarity_for_first_100_bongard_problems()