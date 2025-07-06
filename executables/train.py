import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json
from torchvision.utils import save_image
import time

import sys
sys.path.append('../')

from data.dataset_loader import SentenceToImageDataset
from data.tokeniser import Tokeniser
from model.transformer_model import TextToImageTransformer
from model.loss_functions import PerceptualLoss

TOTAL_EPOCHS = 50
TRAIN_DEBUG = True

def main():
    tokeniser = Tokeniser()
    train_df = pd.read_csv("../data/expanded_sentence_image_relationships.csv")
    tokeniser.build_vocabulary(train_df["sentence"])

    with open("../data/tokeniser_vocab.json", "w") as output:
        json.dump(tokeniser.vocab, output)

    dataset = SentenceToImageDataset("../data/expanded_sentence_image_relationships.csv", tokeniser)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToImageTransformer(vocab_size=len(tokeniser.vocab)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    perceptual_loss = PerceptualLoss().to(device)

    perceptual_loss_weighting = 1.0 # Perceptual loss weighting
    
    config = {
        "vocab_size": len(tokeniser.vocab),
        "embedding_dimensions": 512,
        "total_attention_heads": 16,
        "total_encoder_layers": 8,
        "max_token_size": 64,
        "latent_dim": 128
    }

    with open("../model/config.json", "w") as output:
        json.dump(config, output)

    model.train()
    for epoch in range(TOTAL_EPOCHS):
        print(f"====----Epoch {epoch + 1}----====")
        epoch_start_time = time.perf_counter()
        total_batch_losses = []
        for batch_index, data_batch in enumerate(dataloader):
            batch_start_time = time.perf_counter()
            tokenised_text = data_batch["tokenised_text"].to(device)
            real_left_image = data_batch["left_image"].to(device)
            real_right_image = data_batch["right_image"].to(device)
            raw_sentences_text = data_batch["raw_sentence"]

            predicted_left_image, predicted_right_image = model(tokenised_text)
            p_loss = perceptual_loss(predicted_left_image, real_left_image) + perceptual_loss(predicted_right_image, real_right_image)
            total_batch_loss = perceptual_loss_weighting * p_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_index % 100 == 0:
                print(f"Batch: {batch_index} | Total Batches: {len(dataloader)} | Time Taken for batch: {(time.perf_counter() - batch_start_time):.4f} seconds")

            save_image(real_left_image[:4], "../training_debug/real_left.png", normalize=True)
            save_image(predicted_left_image[:4], "../training_debug/predicted_left.png", normalize=True)
            save_image(real_right_image[:4], "../training_debug/real_right.png", normalize=True)
            save_image(predicted_right_image[:4], "../training_debug/predicted_right.png", normalize=True)            
            total_batch_losses.append(total_batch_loss.detach())

        average_epoch_loss = torch.stack(total_batch_losses).mean().item()
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Average Loss: {average_epoch_loss:.4f} | Time Taken: {(time.perf_counter() - epoch_start_time):.4f} seconds")
        # scheduler.step()
        torch.save(model.state_dict(), "../model/model_weights.pth")
        print("Model weights saved")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main()