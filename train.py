import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json
from torchvision.utils import save_image
import os
import time
from dataset_loader import SentenceToImageDataset
from tokeniser import Tokeniser
from transformer_model import TextToImageTransformer
from loss_functions import CLIPLoss, PerceptualLoss

TOTAL_EPOCHS = 50
TRAIN_DEBUG = True

def main():
    tokeniser = Tokeniser()
    train_df = pd.read_csv("expanded_sentence_image_relationships.csv")
    tokeniser.build_vocabulary(train_df["sentence"])

    with open("tokeniser_vocab.json", "w") as output:
        json.dump(tokeniser.vocab, output)

    dataset = SentenceToImageDataset("expanded_sentence_image_relationships.csv", tokeniser)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToImageTransformer(vocab_size=len(tokeniser.vocab)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    perceptual_loss = PerceptualLoss().to(device)
    clip_loss = CLIPLoss(device).to(device)

    l1_weighting = 0.0 # L1 loss weighting 
    perceptual_loss_weighting = 1.0 # Perceptual loss weighting
    clip_loss_weighting = 1.0 # Clip loss weighting
    
    config = {
        "vocab_size": len(tokeniser.vocab),
        "embedding_dimensions": 512,
        "total_attention_heads": 16,
        "total_encoder_layers": 8,
        "max_token_size": 64
    }

    with open("config.json", "w") as output:
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
            l1 = nn.functional.l1_loss(predicted_left_image, real_left_image) + nn.functional.l1_loss(predicted_right_image, real_right_image)
            p_loss = perceptual_loss(predicted_left_image, real_left_image) + perceptual_loss(predicted_right_image, real_right_image)
            c_loss = clip_loss(predicted_left_image, raw_sentences_text) + clip_loss(predicted_right_image, raw_sentences_text)
            total_batch_loss = l1_weighting * l1 + perceptual_loss_weighting * p_loss + clip_loss_weighting * c_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # try with norm = 5.0 next
            optimizer.step()

            if batch_index % 100 == 0:
                print(f"Batch: {batch_index} | Total Batches: {len(dataloader)} | Time Taken for batch: {(time.perf_counter() - batch_start_time):.4f} seconds")

            real_left_path = os.path.join("training_debug", f"real_left.png")
            predicted_left_path = os.path.join("training_debug", f"predicted_left.png")
            real_right_path = os.path.join("training_debug", f"real_right.png")
            predicted_right_path = os.path.join("training_debug", f"predicted_right.png")

            save_image(real_left_image[:4], real_left_path, normalize=True)
            save_image(predicted_left_image[:4], predicted_left_path, normalize=True)
            save_image(real_right_image[:4], real_right_path, normalize=True)
            save_image(predicted_right_image[:4], predicted_right_path, normalize=True)            
            total_batch_losses.append(total_batch_loss.detach())

        average_epoch_loss = torch.stack(total_batch_losses).mean().item()
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Average Loss: {average_epoch_loss:.4f} | Time Taken: {(time.perf_counter() - epoch_start_time):.4f} seconds")
        # scheduler.step()
        torch.save(model.state_dict(), "model_weights.pth")
        print("Model weights saved")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main()


# Hyperparamters to try in order

#Experiment	λ_l1	λ_p	    λ_c
#Baseline	0.1	    1.0	    1.0	    Current setup
#1	        0.0	    1.0	    0.0	    No CLIP

# clip makes images slightly clearer

# nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
# weight_learner = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
# scheduler = optim.lr_scheduler.StepLR(weight_learner, step_size=20, gamma=0.8)
# scheduler.step()

# Check that CLIP and VGG are actually doing something meaningful