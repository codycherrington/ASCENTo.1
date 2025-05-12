# ASCENTo.1 🚀  
*A personal, terminal-based LLM built from scratch for AI exploration, creative experimentation, and intentional growth.*

---

## 🌟 Highlights

- **Custom Language Model** – Trained from scratch with PyTorch using your own dataset.
- **Transformer Architecture** – Simple, interpretable, and modifiable.
- **Terminal Chat Interface** – Interact with the model directly through your console.
- **Live Training Dashboard** – See loss, perplexity, and progress updates in real time.
- **Session Logging** – Automatically saves loss/perplexity curves and run metadata.
- **Flexible Control Center** – Adjust model dimensions, learning rates, batch sizes, and more.

---

## 🧠 Purpose

ASCENTo.1 was created as a personal learning tool and AI sandbox. It's built to:

- Teach foundational AI architecture hands-on (without relying on black-box APIs).
- Serve as a minimalist LLM you can evolve over time.
- Act as a core framework for future projects, including:
  - Personal growth companions  
  - Creative writing or journaling assistants  
  - AI advisors or bots (financial, mental health, etc.)

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- Matplotlib

---

## ⚙️ Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/codycherrington/ASCENTo.1.git
cd ASCENTo.1
```

### 2. Install dependencies

```bash
pip install torch matplotlib
```

### 3. Launch the model

```bash
python model.py
```

Train from scratch, start a conversation, or expand the dataset.

---

## 📂 Project Structure

```
ASCENTo.1/
├── ascent_data/               # Preprocessed conversations + vocab
├── Archive/                   # Old scripts (vocab builder, etc.)
├── model.py                   # Core model: train, chat, save, etc.
├── dashboard.py               # (Optional) Graphical loss + performance tracking
├── training_logs/             # Logs per training run
├── ascent_tuning_test.txt     # Dev/test samples for quick tuning
├── LICENSE                    # License text
└── README.md                  # This file
```

---

## 💡 Key Features

- 🧠 **EOS-Aware Chat** – Encourages clean sentence endings.
- ♻️ **Repetition Dampening** – Reduces token spam and monotony.
- 🎯 **Top-p + Top-k Sampling** – Flexible, human-like response shaping.
- 📉 **Live Curve Saving** – Loss + perplexity graphs saved after every session.
- 🛠️ **Commented Source** – Everything is fully open and documented.

---

## 📝 License

### MIT License

Copyright (c) 2025 Cody Cherrington

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.**
