import torch
import torch.nn.functional as F
import clip

import sys
sys.path.append('../')

from model.loss_functions import CLIPLoss, PerceptualLoss, normalize_images, convert_to_grayscale, CLIP_MEAN, CLIP_STD
from data.dataset_loader import load_and_transform_image

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
    
def get_cosine_similarity(features1, features2, title):
    features1 = features1.unsqueeze(0)
    features2 = features2.unsqueeze(0)
    cosine_similarity = F.cosine_similarity(features1, features2).item()
    print(f"VGG | {title} | Cosine Similarity: {cosine_similarity}")

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

        # scaler = clip_model.logit_scale.exp()
        cosine_similarity = ((image_embeddings @ text_embeddings.t())).item()
        print(f"CLIP | {title} | Cosine Similarity: {cosine_similarity}")

def clip_similarity_two_images(image_path, image_2_path, title):
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
        
        # scaler = clip_model.logit_scale.exp()
        cosine_similarity = ((image_embeddings @ image_2_embeddings.t())).item()
        print(f"CLIP | {title} | Cosine Similarity: {cosine_similarity}")
        
def get_vgg_cosine_similarity(image_path, image_2_path, title):
    image = load_and_transform_image(IMAGE_SIZE, image_path).to(device)
    image_2 = load_and_transform_image(IMAGE_SIZE, image_2_path).to(device)
    image_features = get_image_features_from_vgg(vgg, image)
    image_2_feautures = get_image_features_from_vgg(vgg, image_2)
    get_cosine_similarity(image_features, image_2_feautures, title)


if __name__ == "__main__":
    bp37_left_image_path = "../bp_images/p037/0.png"
    bp37_right_image_path = "../bp_images/p037/11.png" 

    get_vgg_cosine_similarity(bp37_left_image_path, bp37_right_image_path, "BP37 Opposing Sides")
    clip_similarity_two_images(bp37_left_image_path, bp37_right_image_path, "BP37 Opposing Sides")
    clip_similarity(bp37_left_image_path, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Left Image Correct Text")
    clip_similarity(bp37_right_image_path, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Right Image Correct Text")
    clip_similarity(bp37_left_image_path, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Left Image Wrong Text")
    clip_similarity(bp37_right_image_path, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Right Image Wrong Text")


