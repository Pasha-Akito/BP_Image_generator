import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

class SentenceToImageDataset(Dataset):
    def __init__(self, csv_path, tokeniser, image_size=128):
        self.dataset = pd.read_csv(csv_path)
        self.tokeniser = tokeniser
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        
        token_indices = self.tokeniser.encode(row["sentence"])
        left_image = self.load_and_transform_image(row["left_image"])
        right_image = self.load_and_transform_image(row["right_image"])
    
        return {
            "tokenised_text": torch.tensor(token_indices, dtype=torch.long),
            "left_image": left_image,
            "right_image": right_image
        }    

    def load_and_transform_image(self, image_path):
        image_transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image = Image.open(image_path).convert('L')
        return image_transformer(image)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)