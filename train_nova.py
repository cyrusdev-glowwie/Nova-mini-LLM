import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from transformers_code import Tokenizer, TransformerChat, add_personality, tensorize, PERSONALITY_PROMPT

MAX_LEN = 20
BATCH_SIZE = 16
EPOCHS = 2000
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = [
    ("hi", "hello"),
    ("what is your name", "my name is nova"),
    ("tell me a joke", "why did the computer go to the doctor? because it caught a virus"),
]

all_sentences = [s for pair in pairs for s in pair]
all_sentences.append(PERSONALITY_PROMPT)

tokenizer = Tokenizer()
tokenizer.build_vocab(all_sentences)

model = TransformerChat(tokenizer.vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LR)

def get_batches(pairs, batch_size=BATCH_SIZE):
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        src_batch = [tensorize(add_personality(s[0]))[0] for s in batch]
        trg_batch = [tensorize(s[1])[0] for s in batch]
        src_tensor = torch.stack(src_batch).to(DEVICE)
        trg_tensor = torch.stack(trg_batch).to(DEVICE)
        yield src_tensor, trg_tensor

for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for src, trg in get_batches(pairs, BATCH_SIZE):
        optimizer.zero_grad()
        output = model(src)
        output = output[:, :trg.size(1), :]
        loss = criterion(output.reshape(-1, tokenizer.vocab_size), trg.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "nova_model.pt")
with open("nova_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete! Model and tokenizer saved.")
