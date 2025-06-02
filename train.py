import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json
from torchvision.utils import save_image
import os

from dataset_loader import SentenceToImageDataset
from tokeniser import Tokeniser
from transformer_model import TextToImageTransformer

TOTAL_EPOCHS = 20

def main():
    tokeniser = Tokeniser()
    train_df = pd.read_csv("expanded_sentence_image_relationships.csv")
    tokeniser.build_vocabulary(train_df["sentence"])

    with open("tokeniser_vocab.json", "w") as output:
        json.dump(tokeniser.vocab, output)

    dataset = SentenceToImageDataset("expanded_sentence_image_relationships.csv", tokeniser)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToImageTransformer(vocab_size=len(tokeniser.vocab)).to(device)

    loss_method = nn.L1Loss()
    weight_learner = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.9, eps=1e-8)

    model.train()
    
    for epoch in range(TOTAL_EPOCHS):
        total_epoch_loss = 0.0
        for data_batch in dataloader:
            tokenised_text = data_batch["tokenised_text"].to(device)
            real_left_image = data_batch["left_image"].to(device)
            real_right_image = data_batch["right_image"].to(device)
            
            predicted_left_image, predicted_right_image = model(tokenised_text)
            left_loss = loss_method(predicted_left_image, real_left_image)
            right_loss = loss_method(predicted_right_image, real_right_image)
            total_batch_loss = left_loss + right_loss
            
            weight_learner.zero_grad()
            total_batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            weight_learner.step()

            print("\n=== DEBUG ===")
            print("Real image stats - Left: min={:.3f} max={:.3f} mean={:.3f}".format(real_left_image.min(), real_left_image.max(), real_left_image.mean()))
            print("Pred image stats - Left: min={:.3f} max={:.3f} mean={:.3f}".format(predicted_left_image.min(), real_left_image.max(), predicted_left_image.mean()))
            print("Real image stats - Right: min={:.3f} max={:.3f} mean={:.3f}".format(real_right_image.min(), real_right_image.max(), real_right_image.mean()))
            print("Pred image stats - Right: min={:.3f} max={:.3f} mean={:.3f}".format(predicted_right_image.min(), real_right_image.max(), predicted_right_image.mean()))

            real_left_path = os.path.join("training_debug", f"real_left_.png")
            predicted_left_path = os.path.join("training_debug", f"predicted_left_.png")
            real_right_path = os.path.join("training_debug", f"real_right_.png")
            predicted_right_path = os.path.join("training_debug", f"predicted_right_.png")

            save_image(real_left_image[:4], real_left_path, normalize=True)
            save_image(predicted_left_image[:4], predicted_left_path, normalize=True)
            save_image(real_right_image[:4], real_right_path, normalize=True)
            save_image(predicted_right_image[:4], predicted_right_path, normalize=True)
            print("Saved sample images")
            print("=================\n")
            
            total_epoch_loss += total_batch_loss.item()

        average_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Average Loss: {average_epoch_loss:.4f}")
        torch.save(model.state_dict(), "model_weights.pth")
        print("Model weights saved")
    
    # Save model configuration
    config = {
        "vocab_size": len(tokeniser.vocab),
        "embedding_dimensions": 512,
        "total_attention_heads": 16,
        "total_encoder_layers": 8,
        "max_token_size": 64
    }

    with open("config.json", "w") as output:
        json.dump(config, output)

    


if __name__ == "__main__":
    main()
