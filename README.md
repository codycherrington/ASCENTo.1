# Ascent 1.0 🚀

Welcome to **Ascent** — a personal, lightweight LLM project designed for language experimentation, personal growth, and building an understanding of AI systems from the ground up.

## 🌟 Project Highlights
- **Custom Model:** Built and trained from scratch using PyTorch.
- **Transformer Architecture:** Simple yet powerful self-attention mechanism.
- **Training Dashboard:** Live tracking of loss, perplexity, and progress bars.
- **Conversation Mode:** Chat with your model right inside the terminal.
- **Dataset Expansion:** Easily grow the model's conversational abilities.
- **Training Logs:** Save loss curves, perplexity curves, and run metadata for every session.

## 🚀 Features
- Vocabulary dynamically grows with the dataset.
- Save and reload best-performing models automatically.
- Easy "Control Center" settings to tweak model size, training epochs, batch size, etc.
- Loading bar with estimated time remaining during training.
- Fall-back counter to monitor token generation stability.

## 🛠️ Technologies Used
- Python 3
- PyTorch
- Pandas
- Matplotlib

## 📂 Project Structure
```
Ascent/
├── conversations.py         # Custom conversation data
├── dashboard.py              # Dashboard to monitor training runs
├── model.py                  # Main model file (training, chat, save, etc.)
├── vocab.py                  # Vocabulary building and management
├── progress_reports/         # Progress journals and templates
├── training_logs/            # Saved training session outputs
├── LICENSE                   # MIT License
└── README.md                 # Project overview
```

## ⚙️ Quick Start

Clone the repo:
```
git clone https://github.com/codycherrington/Ascent-1.0.git
```

Install required packages:
```
pip install torch pandas matplotlib
```

Run the project:
```
python model.py
```

Optional: Launch the dashboard separately:
```
python dashboard.py
```

## 🧠 Goals of Ascent
- Learn foundational AI architecture concepts by building hands-on (and with the help of AI tools).
- Create a meaningful companion that improves over time.
- Serve as a flexible skeleton for future projects (trading bots, medical advisors, etc.)

## 📝 License
This project is licensed under the [MIT License](LICENSE).

MIT License

Copyright (c) 2025 Cody Cherrington

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
