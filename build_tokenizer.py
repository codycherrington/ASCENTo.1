import os
import json
import sys


OUTPUT_DIR = "ascent_data"
EOS_TOKEN = "<eos>"

try:
    with open(os.path.join(OUTPUT_DIR, "conversations.json"), "r") as f:
        conversations = json.load(f)
except FileNotFoundError:
    print("❌ conversations.json not found.")
    sys.exit(1)

formatted_convos = []
token_set = set()

for i, convo in enumerate(conversations):
    if "input" not in convo or "output" not in convo:
        print(f"⚠️ Skipping invalid entry at index {i}: {convo}")
        continue

    input_text = convo["input"].strip()
    output_text = convo["output"].strip() + f" {EOS_TOKEN}"
    formatted_convos.append({"input": input_text, "output": output_text})
    token_set.update(input_text.split())
    token_set.update(output_text.split())

with open(os.path.join(OUTPUT_DIR, "conversations.json"), "w") as f:
    json.dump(formatted_convos, f, indent=2)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 2: Create vocab and reverse vocab
sorted_tokens = sorted(token_set)
vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
id_to_word = {str(idx): token for token, idx in vocab.items()}

# Step 3: Save vocab and id_to_word
with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "id_to_word.json"), "w") as f:
    json.dump(id_to_word, f, indent=2)

# Step 4: Save special tokens
special_tokens = {}
if EOS_TOKEN in vocab:
    special_tokens[EOS_TOKEN] = vocab[EOS_TOKEN]

# Add optional <pad> token
PAD_TOKEN = "<pad>"
if PAD_TOKEN not in vocab:
    pad_index = len(vocab)
    vocab[PAD_TOKEN] = pad_index
    id_to_word[str(pad_index)] = PAD_TOKEN
    special_tokens[PAD_TOKEN] = pad_index

with open(os.path.join(OUTPUT_DIR, "special_tokens.json"), "w") as f:
    json.dump(special_tokens, f, indent=2)

print(f"✅ Export complete. Vocab size: {len(vocab)} tokens.")
print(f"📝 Saved files to '{OUTPUT_DIR}': conversations.json, vocab.json, id_to_word.json, special_tokens.json")