# vocab.py

# Master list of tokens (soul words + conversational tokens)
tokens = [
    # Core soul words
    "reflect", "wonder", "become", "shift", "change", "grow", "explore",
    "understand", "realize", "learn", "accept", "journey", "process",
    "unfold", "bloom", "begin", "tend", "nurture", "renew", "restore",
    "align", "balance", "center", "progress", "embrace", "evolve", "adapt",
    "wander", "develop", "deepen", "anchor", "patience", "trust", "stillness",
    "breathe", "surrender", "courage", "resilience", "strength", "persistence",
    "determination", "steady", "foundation", "mindful", "attentive", "observe",
    "soften", "persevere", "start", "finish", "continue", "build", "forge",
    "craft", "weave", "shape", "form", "mold", "reach", "stretch", "climb",
    "rise", "fall", "stand", "overcome", "persist", "recover", "grit",
    "stamina", "pace", "flow", "momentum", "energy", "action", "effort",
    "steps", "direction", "path", "walk", "move", "onward", "curious",
    "dream", "mystery", "question", "ponder", "search", "discover",
    "adventure", "voyage", "expedition", "glimpse", "peek", "insight",
    "spark", "ignite", "illumination", "untangle", "possibility", "openness",
    "clarity", "focus", "vision", "purpose", "meaning", "intention", "goal",
    "plan", "strategy", "resolve", "commit", "dedicate", "aspire", "motivate",
    "inspire", "encourage", "support", "uplift", "empower", "affirm",
    "believe", "hope", "faith", "confidence", "assurance", "security",
    "comfort", "ease", "peace", "calm", "serenity", "tranquility", "relax",
    "release", "letgo", "unwind", "soothe", "gentle", "kind", "compassion",
    "empathy", "care", "connect", "bond", "friendship", "community", "belong",
    "together", "share", "give", "receive", "gratitude", "joy", "delight",
    "happiness", "smile", "laugh", "play", "fun", "celebrate", "cherish",
    "treasure", "value", "honor", "respect", "pride", "selfworth", "selflove",
    "selfcare", "authentic", "genuine", "honest", "truth", "vulnerability",
    "courageous", "brave", "bold", "fearless", "risk", "challenge", "trial",
    "struggle", "adversity", "obstacle", "boundary", "protect", "safe",
    "secure", "pause", "rest", "heal", "mend", "repair", "resolve",
    "content", "abundance", "overflow", "generous", "giving", "gratitude",
    # Conversational tokens (expanded for robustness)
    "hi", "hello", "hey", "friend", "you", "your", "i", "me", "my", "we", "us",
    "they", "them", "it", "this", "that", "these", "those",
    "good", "great", "cool", "nice", "kind", "real", "true", "small", "big",
    "yes", "no", "maybe", "sure", "okay", "alright", "not",
    "please", "thank", "thanks", "sorry", "welcome",
    "feel", "feelings", "thoughts", "think", "know", "understand", "believe",
    "say", "talk", "speak", "listen", "ask", "answer", "share", "learn",
    "can", "could", "will", "would", "should", "must", "might", "try",
    "do", "doing", "did", "done", "make", "made", "build", "create",
    "be", "am", "are", "is", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "help", "support", "guide", "walk", "move", "stay", "leave", "return",
    "begin", "start", "continue", "stop", "end", "finish",
    "happy", "sad", "angry", "scared", "afraid", "excited", "bored",
    "hope", "trust", "patience", "love", "peace", "calm", "quiet", "loud",
    "what", "who", "where", "when", "why", "how", "which",
    "time", "today", "tomorrow", "yesterday", "now", "soon", "later",
    "always", "never", "sometimes", "often", "again", "still",
    "more", "less", "enough", "too", "very", "really",
    "and", "but", "so", "because", "if", "then", "else", "though",
    "before", "after", "around", "through", "over", "under",
    "yes.", "no.", "maybe.", "sure.", "okay.", "bye", "goodbye",
    "see", "hear", "find", "lose", "win", "try", "fail", "grow",
    "important", "meaning", "truth", "life", "light", "dark", "shadow",
    "hopeful", "strong", "brave", "soft", "gentle", "patient", "kindness",
    "trusting", "learning", "curious", "wandering", "becoming", "connected",
]

# Build mapping and reverse lookup
vocab = {word: idx for idx, word in enumerate(tokens)}
vocab_size = len(vocab)
id_to_word = {idx: word for word, idx in vocab.items()}

from conversations import conversations

# Dynamically add any tokens present in conversations but missing from vocab
for pair in conversations:
    if "input" in pair and "output" in pair:
        for w in pair["input"].split() + pair["output"].split():
            if w not in vocab:
                vocab[w] = len(vocab)

# Recompute size and reverse mapping
vocab_size = len(vocab)
id_to_word = {idx: word for word, idx in vocab.items()}