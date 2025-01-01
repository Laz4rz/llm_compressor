import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Compressor:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    def rank_to_token(self, rank, preceeding=None):
        if preceeding is None:
            preceeding = self.tokenizer(" ", return_tensors="pt")
        if isinstance(preceeding, torch.Tensor):
            preceeding = {
                "input_ids": preceeding,
                "attention_mask": torch.ones_like(preceeding)
            }
        if isinstance(preceeding, list):
            preceeding = {
                "input_ids": torch.tensor(preceeding, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(len(preceeding), dtype=torch.long).unsqueeze(0)
            }
        logits = self.model(**preceeding).logits
        ranked = torch.argsort(logits[0, -1, :], descending=True)
        token = ranked[rank].item()
        
        return token
    
    def find_token_rank(self, token, preceeding=None):
        if preceeding is None:
            preceeding = self.tokenizer(" ", return_tensors="pt")
        if isinstance(preceeding, torch.Tensor):
            preceeding = {
                "input_ids": preceeding,
                "attention_mask": torch.ones_like(preceeding)
            }
        if isinstance(preceeding, list):
            preceeding = {
                "input_ids": torch.tensor(preceeding, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(len(preceeding), dtype=torch.long).unsqueeze(0)
            }
        logits = self.model(**preceeding).logits
        ranked = torch.argsort(logits[0, -1, :], descending=True)
        rank = torch.where(ranked == token)[0][0].item()
        return rank
    
    def rank_encode(self, text):
        tokens = self.tokenizer.encode(text)
        ranks = [self.find_token_rank(tokens[0])]
        for i in range(1, len(tokens)):
            preceeding = tokens[:i]
            rank = self.find_token_rank(tokens[i], preceeding)
            ranks.append(rank)
        return ranks

    def rank_decode(self, ranks, return_tokens=False):
        decoded = []
        decoded.append(self.rank_to_token(ranks[0]))
        for i in range(1, len(ranks)):
            preceeding = decoded
            decoded.append(self.rank_to_token(ranks[i], preceeding))
        if return_tokens:
            return decoded
        return self.tokenizer.decode(decoded)

    def _test(self, verbose=False):
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        tokens = self.tokenizer.encode(text)
        ranks = self.rank_encode(text)
        decoded = self.rank_decode(ranks, return_tokens=True)
        decoded_str = self.tokenizer.decode(decoded)
        if verbose:
            print(f"Original text: {text}")
            print(f"Original tokens: {tokens}")
            print(f"Ranks: {ranks}")
            print(f"Tokens: {decoded}")
            print(f"Decoded: {decoded_str}")
        assert text == decoded_str, f"Expected {text}, got {decoded}"

    def encode_with_length_prefix(sequences):
        encoded = ""
        for seq in sequences:
            length = len(seq)
            length_prefix = f"{length:05b}"
            encoded += length_prefix + seq
        return encoded

    def decode_with_length_prefix(encoded):
        sequences = []
        i = 0
        while i < len(encoded):
            # Read the 5-bit length prefix
            length = int(encoded[i:i+5], 2)
            i += 5
            # Extract the sequence
            sequence = encoded[i:i+length]
            sequences.append(sequence)
            i += length
        return sequences
    

if __name__ == "__main__":
    compressor = Compressor()
    compressor._test(verbose=True)
    print("All tests passed.")
    sys.exit(0)