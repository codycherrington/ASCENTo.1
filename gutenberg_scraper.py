

import requests
import re
import json
import os

OUTPUT_PATH = "ascent_data/conversations.json"
BOOKS = {
    "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "Little Women": "https://www.gutenberg.org/files/514/514-0.txt",
    "Anne of Green Gables": "https://www.gutenberg.org/files/45/45-0.txt",
    "The Secret Garden": "https://www.gutenberg.org/files/113/113-0.txt",
    "Peter Pan": "https://www.gutenberg.org/files/16/16-0.txt",
    "Dracula": "https://www.gutenberg.org/files/345/345-0.txt",
    "Dialogues of Plato": "https://www.gutenberg.org/files/1656/1656-0.txt",
    "The Importance of Being Earnest": "https://www.gutenberg.org/files/844/844-0.txt",
    "The Adventures of Sherlock Holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "The Sorrows of Young Werther": "https://www.gutenberg.org/files/2527/2527-0.txt"
}

def extract_conversations(text):
    dialogue_lines = re.findall(r'“([^”]+)”', text)
    convos = []

    for i in range(len(dialogue_lines) - 1):
        a, b = dialogue_lines[i].strip(), dialogue_lines[i + 1].strip()
        if len(a.split()) < 3 or len(b.split()) < 3:
            continue
        convos.append({
            "input": a,
            "output": b,
            "tone": guess_tone(a, b)
        })

    return convos

def guess_tone(input_line, output_line):
    tone = set()

    if any(w in output_line.lower() for w in ["dear", "darling", "love", "child", "sweet", "kind", "please"]):
        tone.add("warm")
    if any(w in output_line.lower() for w in ["perhaps", "you should", "i suggest", "let me", "must", "need to", "ought"]):
        tone.add("helpful")
    if any(w in input_line.lower() for w in ["why", "how", "what", "who", "where", "when"]):
        tone.add("curious")
    if any(w in output_line.lower() for w in ["ha", "funny", "joke", "fool", "nonsense", "ridiculous"]):
        tone.add("playful")
    if "therefore" in output_line.lower() or "because" in output_line.lower():
        tone.add("stoic")
    if not tone:
        tone.add("casual")

    return list(tone)

def scrape_book(title, url):
    print(f"📖 Downloading: {title}")
    res = requests.get(url)
    if res.status_code != 200:
        print(f"❌ Failed to fetch {title}")
        return []
    return extract_conversations(res.text)

if __name__ == "__main__":
    os.makedirs("ascent_data", exist_ok=True)
    all_convos = []

    for title, url in BOOKS.items():
        convos = scrape_book(title, url)
        print(f"✅ {len(convos)} dialogue pairs from {title}")
        all_convos.extend(convos)

    # Load existing conversations if the file exists
    existing_convos = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            existing_convos = json.load(f)

    # Append new conversations
    existing_convos.extend(all_convos)

    # Save updated list
    with open(OUTPUT_PATH, "w") as f:
        json.dump(existing_convos, f, indent=2)

    print(f"\n✨ Appended and saved {len(all_convos)} new conversations to {OUTPUT_PATH}")