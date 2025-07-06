import re
import torch

class Tokeniser:
    def __init__(self):
        self.vocab = {"<PAD>": 0,"<UNK>": 1}

    def build_vocabulary(self, sentences):
        all_tokens = set()

        for sentence in sentences:
            tokens = self.tokenise_sentence(sentence)
            all_tokens.update(tokens)

        for token in sorted(all_tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def tokenise_sentence(self, sentence):
        return re.findall(r"\\?[A-Z]+|\(|\)", sentence)
    
    def encode(self, sentence, max_token_length=64):
        tokens = self.tokenise_sentence(sentence)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Padding
        if (len(token_ids) < max_token_length):
            token_ids += [self.vocab["<PAD>"]] * (max_token_length - len(token_ids))
        else:
            print(f"The sentence '{sentence}' is larger than max_token_length")
            token_ids = token_ids[:max_token_length]

        return token_ids
    
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx_to_token:
                token = self.idx_to_token[token_id]
                if token not in ["<PAD>", "<UNK>"]:  # Skip padding and unknown tokens
                    tokens.append(token)
        
        # Join tokens back into a sentence
        return "".join(tokens)