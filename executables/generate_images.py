import torch
import json
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from PIL import Image
import pandas as pd

import sys
sys.path.append('../')

from model.transformer_model import TextToImageTransformer
from data.tokeniser import Tokeniser
from config import DATASET

TEXT_TO_GENERATE = "Big vs small" 
BP_NUMBER = 2

def force_binary_pixels(images, threshold=0.0):
    # Images are in -1, 1, so threshold can be 0.0
    binary_images = torch.where(images > threshold, torch.ones_like(images), -torch.ones_like(images))
    return binary_images

def adapative_binary_pixels(image):
    # Using image mean as threshold
    threshold = image.mean()
    binary = torch.where(image > threshold, torch.ones_like(image), -torch.ones_like(image))
    return binary

def generate_and_save_images(model, tokeniser, text, device, output_dir="../model_answers"):
    model.eval()
    tokens = tokeniser.encode(text)
    
    with torch.no_grad():
        text_tensor = torch.tensor([tokens]).to(device)
        left, right = model(text_tensor)
    
    left_path = os.path.join(output_dir, "left_temp.png")
    right_path = os.path.join(output_dir, "right_temp.png")

    # left = adapative_binary_pixels(left)
    # right = adapative_binary_pixels(right)
    
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

def generate_and_save_all_image_outputs(model, tokeniser, device):
    bongard_problems_to_infer = pd.read_csv(f"../data/{DATASET}_words_data/bongard_problems_to_test.csv")
    for row in bongard_problems_to_infer.itertuples():
        sentence = row.sentence
        bp_number = row.bp_number
        left_path, right_path = generate_and_save_images(model, tokeniser, sentence, device)    
        create_and_save_plots(left_path, right_path, bp_number, sentence)

def create_and_save_plots(left_path, right_path, bp_number, sentence):
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(left_img)
    ax[0].set_title("Left Image")
    ax[1].imshow(right_img)
    ax[1].set_title("Right Image")
    plt.suptitle(f'BP {bp_number} | "{sentence}"', fontsize=18)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(f'../model_answers_{DATASET}_sentences/BP_{bp_number}_{sentence}.png', dpi=300, bbox_inches='tight')
    print(f'Saved BP_{bp_number}_{sentence}.png')
    plt.close()
    

def main():
    print("Using dataset:", DATASET)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("../model/config.json", "r") as f:
        config = json.load(f)
    
    model = TextToImageTransformer(**config).to(device)
    model.load_state_dict(torch.load("../model/model_weights.pth", map_location=device))
    
    with open("../data/tokeniser_vocab.json", "r") as f:
        vocab = json.load(f)
    tokeniser = Tokeniser()
    tokeniser.vocab = vocab

    # To create and save images for all data found in bongard_problems_to_test.csv
    generate_and_save_all_image_outputs(model, tokeniser, device)

    # To manually create images for a given text and bp 
    # left_path, right_path = generate_and_save_images(model, tokeniser, TEXT_TO_GENERATE, device)
    # create_and_save_plots(left_path, right_path, BP_NUMBER, TEXT_TO_GENERATE)

if __name__ == "__main__":
    main()