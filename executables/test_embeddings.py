import torch
import torch.nn.functional as F
import clip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def clip_similarity(image_path, text, title):
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

        cosine_similarity = ((image_embeddings @ text_embeddings.t())).item()
        print(f"CLIP | {title} | Cosine Similarity: {cosine_similarity}")

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

if __name__ == "__main__":
    bp37_left_image_path = "../bp_images/p037/0.png"
    bp37_right_image_path = "../bp_images/p037/11.png" 

    clip_similarity(bp37_left_image_path, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Left Image Correct Text")
    clip_similarity(bp37_right_image_path, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Right Image Correct Text")
    clip_similarity(bp37_left_image_path, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Left Image Wrong Text")
    clip_similarity(bp37_right_image_path, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Right Image Wrong Text")

    # save_vgg_cosine_similarity_of_first_100_bongard_problems()
    save_clip_cosine_similarity_of_first_100_bongard_problems()