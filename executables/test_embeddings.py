import torch
import torch.nn.functional as F

import sys
sys.path.append('../')

from model.loss_functions import CLIPLoss, PerceptualLoss, normalize_images, convert_to_grayscale
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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    vgg = PerceptualLoss().to(device)

    bp32_left_image = load_and_transform_image(IMAGE_SIZE, "../bp_images/p037/0.png").to(device)
    bp32_right_image = load_and_transform_image(IMAGE_SIZE, "../bp_images/p037/11.png").to(device)

    bp32_left_image_features = get_image_features_from_vgg(vgg, bp32_left_image)
    bp32_right_image_features = get_image_features_from_vgg(vgg, bp32_right_image)

    get_cosine_similarity(bp32_left_image_features, bp32_right_image_features, "BP32 Opposing Sides")


