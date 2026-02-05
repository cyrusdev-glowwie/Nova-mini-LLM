# Nova-mini-LLM
Tiny conversational AI trained with PyTorch (~2-3M parameters)


# Nova - Tiny Conversational AI

Nova is a small mini-LLM chatbot trained on 500+ dialogue pairs using PyTorch. It’s tiny (~2-3M parameters) but fun to chat with!

## Features
- Conversational AI with personality: calm, curious
- Can answer questions, tell jokes, and chat casually
- Tiny transformer model (~2-layer, 128 embedding)
- Open-source and easy to run

## Files
- `transformers_code.py` : Model, tokenizer, chat functions
- `train_nova.py` : Training loop
- `nova_model.pt` : Pretrained model weights (optional)
- `nova_tokenizer.pkl` : Pretrained tokenizer (optional)

## Usage

```python
import torch
from transformers_code import tokenizer, TransformerChat, chat

with open("nova_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = TransformerChat(tokenizer.vocab_size)
model.load_state_dict(torch.load("nova_model.pt"))
model.eval()

print(chat("hi"))
print(chat("what is your name"))
print(chat("tell me a joke"))
