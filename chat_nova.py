import torch
import pickle
from transformers_code import TransformerChat, chat

# Load tokenizer
with open("nova_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = TransformerChat(tokenizer.vocab_size)
model.load_state_dict(torch.load("nova_model.pt", map_location=torch.device('cpu')))
model.eval()

print("Nova is ready to chat! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chat(user_input)
    print(f"Nova: {response}")
