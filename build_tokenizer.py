import os
import json
import sys


OUTPUT_DIR = "ascent_data"
EOS_TOKEN = "<eos>"

try:
    with open(os.path.join(OUTPUT_DIR, "conversations.json"), "r") as f:
        conversations = json.load(f)
except FileNotFoundError:
    print("❌ conversations.json not found. This file is required.")
    sys.exit(1)

# Try to load and combine reddit_conversations.json if present
try:
    with open(os.path.join(OUTPUT_DIR, "reddit_conversations.json"), "r") as f:
        reddit_convos = json.load(f)
        conversations.extend(reddit_convos)
except FileNotFoundError:
    print("⚠️ reddit_conversations.json not found. Continuing without it.")

formatted_convos = []
token_set = set()

for i, convo in enumerate(conversations):
    input_text = convo.get("input", "<empty>").strip()
    output_text = convo.get("output", "<empty>").strip()

    # Normalize and skip if clearly junk
    if (
        input_text.lower() == "[removed]"
        or len(input_text) < 5
        or all(char in "!?.<>" for char in input_text)
        or input_text == ""
    ):
        print(f"⚠️ Skipping removed or junk input at index {i}")
        continue
    if (
        output_text.lower().startswith("[removed]")
        or len(output_text) < 5
        or all(char in "!?.<>" for char in output_text)
        or output_text == ""
    ):
        print(f"⚠️ Skipping removed or junk output at index {i}")
        continue

    output_text += f" {EOS_TOKEN}"
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

# Add optional <pad> token
PAD_TOKEN = "<pad>"
if PAD_TOKEN not in vocab:
    pad_index = len(vocab)
    vocab[PAD_TOKEN] = pad_index
    id_to_word[str(pad_index)] = PAD_TOKEN

# Add <unk> token
UNK_TOKEN = "<unk>"
if UNK_TOKEN not in vocab:
    unk_index = len(vocab)
    vocab[UNK_TOKEN] = unk_index
    id_to_word[str(unk_index)] = UNK_TOKEN

# Step 3: Save vocab and id_to_word
with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "id_to_word.json"), "w") as f:
    json.dump(id_to_word, f, indent=2)

 # Step 4: Save special tokens
special_tokens = {}
if EOS_TOKEN in vocab:
    special_tokens[EOS_TOKEN] = vocab[EOS_TOKEN]
if PAD_TOKEN in vocab:
    special_tokens[PAD_TOKEN] = vocab[PAD_TOKEN]
if UNK_TOKEN in vocab:
    special_tokens[UNK_TOKEN] = vocab[UNK_TOKEN]

with open(os.path.join(OUTPUT_DIR, "special_tokens.json"), "w") as f:
    json.dump(special_tokens, f, indent=2)

print(f"✅ Export complete. Vocab size: {len(vocab)} tokens.")
print(f"📝 Saved files to '{OUTPUT_DIR}': conversations.json, vocab.json, id_to_word.json, special_tokens.json")