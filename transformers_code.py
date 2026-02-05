import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2}
        self.idx2word = {0:"<pad>", 1:"<sos>", 2:"<eos>"}
        self.vocab_size = 3

    def build_vocab(self, sentences):
        idx = len(self.word2idx)
        for sentence in sentences:
            for w in sentence.split():
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    idx +=1
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):
        return [self.word2idx["<sos>"]] + [self.word2idx[w] for w in sentence.split()] + [self.word2idx["<eos>"]]

    def decode(self, tokens):
        words = [self.idx2word[t] for t in tokens if t not in [self.word2idx["<sos>"], self.word2idx["<eos>"], self.word2idx["<pad>"]]]
        return " ".join(words)

class TransformerChat(nn.Module):
    def __init__(self, vocab_size, embed_size=128, nhead=4, num_layers=2, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, 20, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

MAX_LEN = 20

def pad_sequence(tokens):
    if len(tokens) < MAX_LEN:
        tokens += [tokenizer.word2idx["<pad>"]] * (MAX_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_LEN]
    return tokens

def tensorize(sentence):
    tokens = tokenizer.encode(sentence)
    tokens = pad_sequence(tokens)
    return torch.tensor([tokens], dtype=torch.long)

PERSONALITY_PROMPT = "i am an ai"

def add_personality(text):
    return PERSONALITY_PROMPT + " " + text

def chat(input_text, temperature=0.5, max_len=20):
    model.eval()
    with torch.no_grad():
        src = tensorize(add_personality(input_text))
        output = model(src)
        tokens = []
        for t in range(max_len):
            probs = torch.softmax(output[0, t] / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == tokenizer.word2idx["<eos>"]:
                break
            tokens.append(next_token)
        return tokenizer.decode(tokens)
