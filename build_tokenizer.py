import os
import json
import sys
import re


OUTPUT_DIR = "ascent_data"
EOS_TOKEN = "<eos>"

# Load primary conversations only for saving
primary_conversations = []
try:
    with open(os.path.join(OUTPUT_DIR, "conversations.json"), "r") as f:
        primary_conversations = json.load(f)
except FileNotFoundError:
    print("⚠️ conversations.json not found. Skipping.")

# Load all for tokenizer input
conversations = list(primary_conversations)

try:
    with open(os.path.join(OUTPUT_DIR, "curated_conversations.json"), "r") as f:
        conversations.extend(json.load(f))
except FileNotFoundError:
    print("⚠️ curated_conversations.json not found. Skipping.")

try:
    with open(os.path.join(OUTPUT_DIR, "identity.json"), "r") as f:
        conversations.extend(json.load(f))
except FileNotFoundError:
    print("⚠️ identity.json not found. Skipping.")

if not conversations:
    print("❌ No valid conversation files found.")
    sys.exit(1)

def normalize(text):
    # Remove EOS_TOKEN, normalize whitespace, lowercase
    return re.sub(r'\s+', ' ', text.replace(EOS_TOKEN, "")).strip().lower()

def signature(convo):
    return (normalize(convo.get("input", "")), normalize(convo.get("output", "")))

duplicate_signatures = set()

for filename in ["curated_conversations.json", "identity.json"]:
    try:
        with open(os.path.join(OUTPUT_DIR, filename), "r") as f:
            for entry in json.load(f):
                duplicate_signatures.add(signature(entry))
    except FileNotFoundError:
        continue

# Filter primary conversations before formatting
filtered_raw_primary = [
    c for c in primary_conversations
    if signature(c) not in duplicate_signatures
]

token_set = set()

def is_junk(text):
    text = text.strip().lower()
    return (
        text in ["", "[removed]"]
        or "[deleted]" in text
        or len(text) < 5
        or all(char in "!?.<>" for char in text)
    )

# Now re-format only the filtered primary conversations
formatted_convos = []
for i, convo in enumerate(filtered_raw_primary):
    input_text = convo.get("input", "<empty>").strip()
    output_text = convo.get("output", "<empty>").strip()

    input_text = re.sub(r'\s+([.,!?;:])', r'\1', input_text)
    output_text = re.sub(r'\s+([.,!?;:])', r'\1', output_text)

    if is_junk(input_text) or is_junk(output_text):
        print(f"⚠️ Skipping removed or junk line from original conversations.")
        continue

    output_text = output_text.replace(f"{EOS_TOKEN}", "").strip() + f" {EOS_TOKEN}"
    convo_data = {"input": input_text, "output": output_text}
    if "tone" in convo:
        convo_data["tone"] = convo["tone"]
    formatted_convos.append(convo_data)
    token_set.update(input_text.split())
    token_set.update(output_text.split())

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

# Overwrite conversations.json with cleaned, non-duplicate entries
with open(os.path.join(OUTPUT_DIR, "conversations.json"), "w") as f:
    json.dump(formatted_convos, f, indent=2)

print(f"✅ Export complete. Vocab size: {len(vocab)} tokens.")
print(f"📝 Tokenizer processed input from: conversations.json, curated_conversations.json, identity.json")
print(f"📦 Saved to '{OUTPUT_DIR}': conversations.json (cleaned), vocab.json, id_to_word.json, special_tokens.json")