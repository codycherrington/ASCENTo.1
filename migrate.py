# migrate_conversations.py

import json
import os
from conversations import conversations

# Ensure target directory exists
os.makedirs("ascent_data", exist_ok=True)

# Define output path
output_path = os.path.join("ascent_data", "conversations.json")

# Write the conversations to a JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(conversations)} conversations to {output_path}")