import torch
import torch.nn as nn
from torchvision import models
import clip

def normalize_images(images, mean, std):
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor

def convert_to_grayscale(images):
    return images.repeat(1, 3, 1, 1)

class PerceptualLoss(nn.Module):    
    # VGG layers for feature extraction
    FEATURE_LAYERS = [2, 7, 12, 21, 30]
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
        predicted = convert_to_grayscale(predicted)
        real = convert_to_grayscale(real)
        
        predicted = normalize_images(predicted, self.VGG_MEAN, self.VGG_STD)
        real = normalize_images(real, self.VGG_MEAN, self.VGG_STD)
        
        total_loss = 0.0
        for feature_extractor in self.feature_blocks:
            predicted = feature_extractor(predicted)
            real = feature_extractor(real)
            total_loss += self.mse_loss(predicted, real)
            
        return total_loss

class CLIPLoss(nn.Module):
    # CLIP normalisation parameters https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, predicted, real):
        predicted = convert_to_grayscale(predicted)
        real = convert_to_grayscale(real)
        combined_batch = torch.cat([predicted, real], dim=0)
        
        # Resizing to fit CLIP
        resized = nn.functional.interpolate(combined_batch, size=(224, 224), mode='bilinear', align_corners=False)
        normalized_images = normalize_images(resized, self.CLIP_MEAN, self.CLIP_STD)

        # Generate Embeddings
        embeddings = self.clip_model.encode_image(normalized_images)
        predicted_embeddings, real_embeddings = torch.split(embeddings, predicted.size(0), dim=0)
        
        # Calculate cosine similarity loss
        cosine_similarity = nn.functional.cosine_similarity(predicted_embeddings, real_embeddings, dim=-1)
        return (1 - cosine_similarity).mean()