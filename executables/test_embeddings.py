import torch
import torch.nn.functional as F
import clip

import sys
sys.path.append('../')

from model.loss_functions import CLIPLoss, PerceptualLoss, normalize_images, convert_to_grayscale, CLIP_MEAN, CLIP_STD
from data.dataset_loader import load_and_transform_image

IMAGE_SIZE = 128

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

def clip_similarity(clip_model, image, text, title):

    trained_clip_state = torch.load("../model/clip_projection_weights.pth", map_location=next(clip_model.clip_model.parameters()).device)
    clip_model.image_proj.load_state_dict(trained_clip_state['image_proj'])
    clip_model.text_proj.load_state_dict(trained_clip_state['text_proj'])

    image = image.unsqueeze(0)
    resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    gray_scale = convert_to_grayscale(resized)
    norm_image = normalize_images(gray_scale, CLIP_MEAN, CLIP_STD)
    
    with torch.no_grad():
        image_embeddings = clip_model.clip_model.encode_image(norm_image)
        text_tokens = clip.tokenize([text], truncate=True).to(next(clip_model.parameters()).device)
        text_embeddings = clip_model.clip_model.encode_text(text_tokens)

        image_embeddings = clip_model.image_proj(image_embeddings)
        text_embeddings = clip_model.text_proj(text_embeddings)

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        scaler = clip_model.clip_model.logit_scale.exp()
        cosine_similarity = (scaler * (image_embeddings @ text_embeddings.t())).item()

        print(f"CLIP | {title} | Cosine Similarity: {cosine_similarity}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = PerceptualLoss().to(device)
    clip_model = CLIPLoss(device).to(device)

    bp37_left_image = load_and_transform_image(IMAGE_SIZE, "../bp_images/p037/0.png").to(device)
    bp37_right_image = load_and_transform_image(IMAGE_SIZE, "../bp_images/p037/11.png").to(device)

    bp37_left_image_features = get_image_features_from_vgg(vgg, bp37_left_image)
    bp37_right_image_features = get_image_features_from_vgg(vgg, bp37_right_image)

    get_cosine_similarity(bp37_left_image_features, bp37_right_image_features, "BP37 Opposing Sides")

    clip_similarity(clip_model, bp37_left_image, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Left Image Correct Text")
    clip_similarity(clip_model, bp37_right_image, "LEFT(GREATERLLA(TRIANGLES,CIRCLES,YPOS))", "BP37 Right Image Correct Text")

    clip_similarity(clip_model, bp37_left_image, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Left Image Wrong Text")
    clip_similarity(clip_model, bp37_right_image, "RIGHT(LESSSIMLA(FIGURES,SIZE))", "BP37 Right Image Wrong Text")