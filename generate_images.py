import torch
from transformer_model import TextToImageTransformer
from tokeniser import Tokeniser
import json
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from datetime import datetime
from PIL import Image

TEST_GENERATOR = True
# TEXT_TO_GENERATE = "\LEFT(\GREATERLA(\FIGURES,\SIZE))"
TEXT_TO_GENERATE = "\LEFT(\MORE(\FIGURES,\SIZE))"

def generate_and_save_images(model, tokeniser, text, device, output_dir="outputs"):
    model.eval()
    tokens = tokeniser.encode(text)
    
    with torch.no_grad():
        text_tensor = torch.tensor([tokens]).to(device)
        left, right = model(text_tensor)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    left_path = os.path.join(output_dir, f"left_{timestamp}.png")
    right_path = os.path.join(output_dir, f"right_{timestamp}.png")
    
    save_image(left, left_path, normalize=True)
    save_image(right, right_path, normalize=True)
    
    return left_path, right_path

def test_generator(generator, device):
    test_noise = torch.randn(1, 512).to(device) 
    fake_image = generator(test_noise)
    
    print("Generator output shape:", fake_image.shape)
    print("Min value:", fake_image.min().item())
    print("Max value:", fake_image.max().item())
    print("Mean value:", fake_image.mean().item())
    
    plt.imshow(fake_image[0,0].detach().cpu().numpy(), cmap='gray')
    plt.title("Generator Test Output")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model = TextToImageTransformer(**config).to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))

    if TEST_GENERATOR:
        test_generator(model.left_generator, device)
    
    with open("tokeniser_vocab.json", "r") as f:
        vocab = json.load(f)
    tokeniser = Tokeniser()
    tokeniser.vocab = vocab

    left_path, right_path = generate_and_save_images(model, tokeniser, TEXT_TO_GENERATE, device)    

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    left_img = Image.open(left_path)
    right_img = Image.open(right_path)
    
    ax[0].imshow(left_img)
    ax[0].set_title("Left Image")
    ax[1].imshow(right_img)
    ax[1].set_title("Right Image")
    plt.suptitle(f'Input: "{TEXT_TO_GENERATE}"')
    plt.show()


if __name__ == "__main__":
    main()