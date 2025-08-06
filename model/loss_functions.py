import torch
import torch.nn as nn
from torchvision import models
import clip

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def normalize_images(images, mean, std):
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor

def convert_to_grayscale(images):
    return images.repeat(1, 3, 1, 1)

class PerceptualLoss(nn.Module):    
    # VGG layers for feature extraction
    FEATURE_LAYERS = [2, 4, 7, 9, 12, 14, 16, 21]
    # LAYER_WEIGHTS = [1.8, 1.4, 0.9, 0.7, 0.5, 0.35, 0.25, 0.15] # getting more accurate images for abstract concepts from this weighting
    LAYER_WEIGHTS = [1.9, 1.5, 1.0, 0.8, 0.5, 0.35, 0.25, 0.15] 
    # LAYER_WEIGHTS = [1.8, 1.6, 1.2, 1.0, 0.6, 0.4, 0.3, 0.2] # try this next 
    # LAYER_WEIGHTS = [2.0, 1.6, 1.1, 0.6, 0.45, 0.35, 0.25, 0.15] # Try decreasing the middle weights a little


    # VGG normalization parameters print(torchvision.models.VGG19_Weights.DEFAULT.transforms())
    VGG_MEAN = [0.485, 0.456, 0.406]
    VGG_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        for layer in vgg: # Inplace for Relu to not break
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        vgg.eval() # Freezing layers
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature blocks at specified layers
        self.feature_blocks = nn.ModuleList()
        start_idx = 0
        for end_idx in self.FEATURE_LAYERS:
            block = nn.Sequential(*vgg[start_idx:end_idx + 1])
            self.feature_blocks.append(block.eval())
            start_idx = end_idx + 1
        
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, real):
        # converting tanh range into 0,1 for VGG
        predicted = (predicted + 1) / 2 
        real = (real + 1) / 2 

        predicted = convert_to_grayscale(predicted)
        real = convert_to_grayscale(real)
        
        predicted = normalize_images(predicted, self.VGG_MEAN, self.VGG_STD)
        real = normalize_images(real, self.VGG_MEAN, self.VGG_STD)
        
        total_loss = 0.0
        for i, feature_extractor in enumerate(self.feature_blocks):
            predicted = feature_extractor(predicted)
            real = feature_extractor(real)
            total_loss += self.LAYER_WEIGHTS[i] * self.mse_loss(predicted, real)
            
        return total_loss

class CLIPLoss(nn.Module):
    # CLIP normalisation parameters https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
    def __init__(self, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, generated_images, text_descriptions):
        # https://github.com/openai/CLIP/blob/main/clip/model.py#L358
        resized = nn.functional.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False)
        gray_scale_images = convert_to_grayscale(resized)
        norm_images = normalize_images(gray_scale_images, CLIP_MEAN, CLIP_STD)
        
        text_tokens = clip.tokenize(text_descriptions, truncate=True).to(generated_images.device)
        
        image_embeddings = self.clip_model.encode_image(norm_images)
        text_embeddings = self.clip_model.encode_text(text_tokens)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        scaler = self.clip_model.logit_scale.exp()
        cosine_similarities = scaler * (image_embeddings @ text_embeddings.t())
        
        targets = torch.arange(len(generated_images), device=generated_images.device)
        image_loss = nn.functional.cross_entropy(cosine_similarities, targets)
        text_loss = nn.functional.cross_entropy(cosine_similarities.t(), targets)
        return (image_loss + text_loss) / 2
