import torch
import torch.nn as nn

class TextToImageTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dimensions=256, total_attention_heads=8, total_encoder_layers=4, max_token_size=64):
        super().__init__()

        # Embedding 
        self.embedding = nn.Embedding(vocab_size, embedding_dimensions) 
        self.position_embedding = nn.Embedding(max_token_size, embedding_dimensions) 

        # Transformer Architecture using pytorch
        single_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dimensions, nhead=total_attention_heads)
        self.transformer_encoder = nn.TransformerEncoder(single_encoder_layer, num_layers=total_encoder_layers) 

        # Split Embedding for dual image generators
        self.split_embedding = nn.Linear(embedding_dimensions, embedding_dimensions * 2)

        self.left_generator = self.build_generator(embedding_dimensions)
        self.right_generator = self.build_generator(embedding_dimensions)

    def build_generator(self, embedding_dimensions):
        return nn.Sequential(
        nn.Linear(embedding_dimensions, 512 * 8 * 8),
        nn.Unflatten(1, (512, 8, 8)),
        nn.ConvTranspose2d(512, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
        )
    
    def forward(self, text):        
        # Creating embeddings 
        position_indices = torch.arange(0, text.size(1)).to(text.device)
        position_embeddings = self.position_embedding(position_indices)
        token_embeddings = self.embedding(text)
        combined_embeddings = token_embeddings + position_embeddings

        transformer_output = self.transformer_encoder(combined_embeddings)
        context_vector = transformer_output.mean(dim=1)

        # Split and generate images
        split_context_vector = self.split_embedding(context_vector)
        left_embedding, right_embedding = split_context_vector.chunk(2, dim=1)
        left_image = self.left_generator(left_embedding)
        right_image = self.right_generator(right_embedding)
        return left_image, right_image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)