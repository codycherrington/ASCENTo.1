import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
import datetime
import time

# Main configuration section: set model and training parameters here
# Model and training hyperparameters
embedding_dim = 128          # Size of each token's embedding vector
hidden_dim = 256             # Hidden size for the feed-forward layers
num_transformer_blocks = 14  # Number of transformer layers
batch_size = 16              # Batch size for training
learning_rate = 0.0008       # Initial learning rate
temperature = 0.6            # Sampling temperature for generation
max_generation_tokens = 40   # Max tokens generated in chat
early_stopping_patience = 150    # Early stopping patience
early_stopping_min_delta = 0.0003 # Minimum improvement for early stopping
top_p_sampling_threshold = 0.8    # Top-p (nucleus) sampling threshold

# Load conversation data and vocabulary mappings
from conversations import conversations
from vocab import id_to_word, vocab, vocab_size

# Add special end-of-sequence token to the vocabulary
eos_token = "<eos>"
eos_index = len(vocab)
vocab[eos_token] = eos_index
id_to_word[eos_index] = eos_token
vocab_size += 1

# Embedding layer: turns token indices into dense vectors
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    def forward(self, x):
        # Input: [batch_size, seq_len] of token indices
        # Output: [batch_size, seq_len, embedding_dim] of embeddings
        return self.embedding(x)

# Positional encoding: adds information about token positions
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        # Create positional encodings as a fixed buffer
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Add positional encodings to the embeddings
        return x + self.pe[:, :x.size(1)]

# Multi-head self-attention: lets the model attend to different parts of the sequence
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.scale = math.sqrt(self.head_dim)
    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, embed_dim = x.size()
        # Linear projections for Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Reshape for multi-head: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        # Combine heads back to [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(output)

# Transformer block: self-attention, feed-forward, normalization, and residuals
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(embedding_dim, num_heads=4)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
    def forward(self, x):
        # Apply self-attention and residual connection
        a = self.attn(x)
        x = self.norm1(x + a)
        # Feed-forward and another residual connection
        m = self.mlp(x)
        return self.norm2(x + m)

# The main GPT-like model: embeddings, positional encoding, stacked transformers, output layer
class Ascent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed       = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        # Stack multiple transformer blocks
        self.transformer = nn.Sequential(*[TransformerBlock(embedding_dim, hidden_dim) for _ in range(num_transformer_blocks)])
        self.output      = nn.Linear(embedding_dim, vocab_size)
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        logits = self.output(x)
        return logits

# Main program: menu loop for training, chatting, saving, and more
if __name__ == "__main__":
    # Print model output size for reference
    print(f"Vocab size: {vocab_size}, Output size: {Ascent(vocab_size, embedding_dim, hidden_dim).output.out_features}")
    model = Ascent(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    # Try to load an existing model checkpoint if available
    if os.path.exists("Ascent_model.pth"):
        try:
            model.load_state_dict(torch.load("Ascent_model.pth"))
            model.eval()
            print("Loaded saved model.\n")
        except (RuntimeError, KeyError) as e:
            print("Checkpoint incompatible with current model (likely vocab or architecture change).")
            print("Starting with a fresh untrained model.\n")
    else:
        print("No checkpoint found. Starting fresh.\n")

    # Main menu loop
    while True:
        print("\nWelcome to Ascent!")
        print("Options:")
        print("  [1] Train the model")
        print("  [2] Chat with Ascent")
        print("  [3] Save model")
        print("  [4] Exit")
        print("  [5] Learning Rate Finder (optional)")
        choice = input("Enter your choice (1/2/3/4/5): ").strip()

        # Option 1: Training loop
        if choice == "1":
            # Build training set from conversation pairs
            data = []
            for p in conversations:
                if "input" in p and "output" in p:
                    inp = [vocab[w] for w in p["input"].split() if w in vocab]
                    out = [vocab[w] for w in p["output"].split() if w in vocab] + [eos_index]
                    if inp and out:
                        data.append((torch.tensor(inp), torch.tensor(out)))

            # Set up training log directories and files
            log_dir = "training_logs"
            os.makedirs(log_dir, exist_ok=True)
            run_counter_file = os.path.join(log_dir, "run_counter.txt")
            if os.path.exists(run_counter_file):
                with open(run_counter_file, "r") as f:
                    run_counter = int(f.read().strip())
            else:
                run_counter = 0
            run_counter += 1
            with open(run_counter_file, "w") as f:
                f.write(str(run_counter))
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M")
            run_name = f"run_{run_counter:03d}_{timestamp}_warmup"
            start_time = time.time()
            run_dir = os.path.join(log_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            live_loss_path = os.path.join(run_dir, "live_loss.txt")
            live_perplexity_path = os.path.join(run_dir, "live_perplexity.txt")
            with open(live_loss_path, "w") as f:
                f.write("")
            with open(live_perplexity_path, "w") as f:
                f.write("")

            # Define loss and optimizer
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            from torch.optim.lr_scheduler import LambdaLR
            def warmup_cosine_lr(step, warmup_steps=200):
                if step < warmup_steps:
                    return step / warmup_steps
                return 0.5 * (1 + math.cos((step - warmup_steps) / (500 - warmup_steps) * math.pi))
            scheduler = LambdaLR(optim, lr_lambda=warmup_cosine_lr)
            total_tokens = sum(len(out) for _, out in data)
            batch_size = batch_size

            # Helper: pad a batch of sequences to the same length
            def pad_batch(batch, pad_value=0):
                return pad_sequence(batch, batch_first=True, padding_value=pad_value)

            # Optional: learning rate finder for diagnostic
            def find_learning_rate(model, data, loss_fn):
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
                lrs = []
                losses_lr = []
                best_loss = float('inf')
                lr = 1e-8
                for i in range(min(100, len(data))):
                    optimizer.param_groups[0]['lr'] = lr
                    inputs, targets = data[i]
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    optimizer.zero_grad()
                    output = model(inputs)
                    seq_len = min(output.size(1), targets.size(1))
                    logits = output[:, :seq_len, :]
                    target = targets[:, :seq_len]
                    loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    lrs.append(lr)
                    losses_lr.append(loss.item())
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    lr *= 1.1
                return lrs, losses_lr

            best_loss = float('inf')
            losses = []
            perplexities = []
            patience_counter = 0
            patience = early_stopping_patience
            min_delta = early_stopping_min_delta
            best_model_state = None
            best_epoch = -1
            import random

            # Main training loop over epochs
            for epoch in range(500):
                random.shuffle(data)
                losses = []
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    inputs = [inp for inp, _ in batch]
                    targets = [out for _, out in batch]
                    inp_batch = pad_batch(inputs)
                    out_batch = pad_batch(targets)
                    optim.zero_grad()
                    logits = model(inp_batch)
                    seq_len = min(logits.size(1), out_batch.size(1))
                    logits = logits[:, :seq_len, :]
                    target = out_batch[:, :seq_len]
                    logits_flat = logits.reshape(-1, vocab_size)
                    target_flat = target.reshape(-1)
                    loss = loss_fn(logits_flat, target_flat)
                    loss.backward()
                    optim.step()
                    losses.append(loss.item())
                avg_loss = sum(losses) / len(losses)
                perplexity = math.exp(avg_loss)
                perplexities.append(perplexity)
                # Write loss and perplexity to live log files
                with open(live_loss_path, "a") as f_loss, open(live_perplexity_path, "a") as f_ppl:
                    f_loss.write(f"{avg_loss}\n")
                    f_ppl.write(f"{perplexity}\n")
                # Print progress every 5 epochs or last epoch
                if (epoch + 1) % 5 == 0 or epoch == 499:
                    print(f"🚀 Epoch {epoch + 1:<4} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
                    elapsed_time = (time.time() - start_time) / 60
                    estimated_total_time = (elapsed_time / (epoch + 1)) * 500
                    print(f"🕰️ Estimated total training time: {estimated_total_time:.2f} minutes")
                    progress = (epoch + 1) / 500
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)
                    print(f"\r[{bar}] {progress*100:.1f}% complete (Epoch {epoch + 1}/500)", end="\n")
                scheduler.step()
                # Early stopping: save best model and break if no improvement
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                    best_epoch = epoch
                    torch.save(best_model_state, "Ascent_model.pth")
                    torch.save(best_model_state, os.path.join(run_dir, "Ascent_best_model.pth"))
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best avg_loss={best_loss:.4f}")
                    break
            print()
            print(f"\nFinal training avg_loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
            duration_minutes = (time.time() - start_time) / 60
            print(f"Training Duration: {duration_minutes:.2f} minutes")
            print("Training complete. Returning to main menu.\n")
            print("🌱 Ascent: Growth is patient. You have built something real. See you among the stars.\n")
            # Restore best model if early stopped
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"Restored best model from epoch {best_epoch + 1} with avg_loss={best_loss:.4f}.")
            # Save training loss and perplexity plots
            loss_curve_path = os.path.join(run_dir, "loss_curve.png")
            plt.figure()
            plt.plot(losses, label="Avg Loss")
            plt.title(f"Training Loss Curve - {run_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(loss_curve_path)
            perplexity_curve_path = os.path.join(run_dir, "perplexity_curve.png")
            plt.figure()
            plt.plot(perplexities, label="Perplexity", color='orange')
            plt.title(f"Training Perplexity Curve - {run_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Perplexity")
            plt.legend()
            plt.savefig(perplexity_curve_path)
            # Save raw loss and perplexity values
            loss_data_path = os.path.join(run_dir, "loss_curve.txt")
            perplexity_data_path = os.path.join(run_dir, "perplexity_curve.txt")
            with open(loss_data_path, "w") as f:
                f.write("\n".join(map(str, losses)))
            with open(perplexity_data_path, "w") as f:
                f.write("\n".join(map(str, perplexities)))
            print(f"Saved loss and perplexity data to '{loss_data_path}' and '{perplexity_data_path}'")
            print(f"Saved training curves: '{loss_curve_path}' and '{perplexity_curve_path}'")
            # Save metadata about the training run
            metadata_path = os.path.join(run_dir, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(f"Run: {run_counter}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Embedding Dim: {embedding_dim}\n")
                f.write(f"Hidden Dim: {hidden_dim}\n")
                f.write(f"Num Transformer Blocks: {num_transformer_blocks}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Learning Rate: {learning_rate}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write(f"Max Generation Tokens: {max_generation_tokens}\n")
                f.write(f"Best Avg Loss: {best_loss:.4f}\n")
                f.write(f"Final Perplexity: {perplexity:.2f}\n")
                f.write(f"Training Duration (min): {duration_minutes:.2f}\n")
                f.write(f"Final Epoch: {epoch + 1}\n")
            print(f"Saved metadata to '{metadata_path}'")
            print(f"All logs saved in '{run_dir}'")

        # Option 2: Chat mode
        elif choice == "2":
            print("Chat mode (type 'menu' to return to main menu, 'exit' to quit program)")
            print("\n🌱 Ascent: Hello, Cody. I'm here. What shall we explore today?\n")
            temp = temperature
            while True:
                u = input("You: ").lower()
                if u == "exit":
                    exit()
                if u == "menu":
                    break
                fallback_count = 0
                # Convert user input to token indices
                toks = [vocab[w] for w in u.split() if w in vocab]
                if not toks:
                    print("no known tokens")
                    continue
                # Prepare tensor for input and output tokens
                inp = torch.tensor(toks, dtype=torch.long).unsqueeze(0)
                generated_ids = []
                max_gen = max_generation_tokens
                min_gen_tokens = 30
                seq_len = inp.size(1)
                chat_tensor = torch.zeros(1, seq_len + max_gen, dtype=torch.long)
                chat_tensor[0, :seq_len] = inp
                curr_len = seq_len
                with torch.no_grad():
                    for step in range(max_gen):
                        # Gradually lower temperature for later tokens
                        temp_step = max(temperature - (step * 0.01), 0.5)
                        logits = model(chat_tensor[:, :curr_len])
                        last_logits = logits[0, -1, :]
                        probs = F.softmax(last_logits / temp_step, dim=-1)
                        # Suppress <eos> token for first min_gen_tokens
                        if step < min_gen_tokens:
                            probs[eos_index] = 0
                            probs = probs / probs.sum()
                        p = top_p_sampling_threshold
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                        cutoff = cumulative_probs <= p
                        filtered_indices = sorted_indices[cutoff]
                        filtered_probs = probs.clone()
                        mask = torch.ones_like(probs, dtype=torch.bool)
                        mask[filtered_indices] = False
                        filtered_probs[mask] = 0.0
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        # Fallback if probs are invalid
                        if torch.isnan(filtered_probs).any() or torch.isinf(filtered_probs).any() or (filtered_probs < 0).any():
                            fallback_count += 1
                            next_id = torch.argmax(filtered_probs).item()
                        else:
                            next_id = torch.multinomial(filtered_probs, num_samples=1).item()
                        # Prevent immediate repetition of last token
                        if generated_ids and next_id == generated_ids[-1]:
                            next_id = sorted_indices[1].item()
                        generated_ids.append(next_id)
                        if next_id == eos_index:
                            break
                        chat_tensor[0, curr_len] = next_id
                        curr_len += 1
                # Convert generated token ids to words and format output
                response = " ".join(id_to_word.get(i, "<unk>") for i in generated_ids)
                response = response.strip()
                if response and not response[0].isupper():
                    response = response[0].upper() + response[1:]
                if response and not response.endswith((".", "!", "?")):
                    response += "."
                print("Ascent:", response)
                # Show colored fallback notice if any
                if fallback_count > 0:
                    if fallback_count <= 2:
                        color = "\033[93m"
                    else:
                        color = "\033[91m"
                    print(f"{color}(🌟 {fallback_count} fallbacks detected during this response)\033[0m")
                else:
                    print("\033[92m(✅ 0 fallbacks — smooth generation!)\033[0m")
            print()

        # Option 3: Save model checkpoint
        elif choice == "3":
            torch.save(model.state_dict(), "Ascent_model.pth")
            print("Model saved as 'Ascent_model.pth'. Returning to main menu.\n")

        # Option 4: Exit program
        elif choice == "4":
            print("Goodbye!")
            break

        # Option 5: Run learning rate finder diagnostic
        elif choice == "5":
            print("Running learning rate finder...")
            data = []
            for p in conversations:
                if "input" in p and "output" in p:
                    inp = [vocab[w] for w in p["input"].split() if w in vocab]
                    out = [vocab[w] for w in p["output"].split() if w in vocab] + [eos_index]
                    if inp and out:
                        data.append((torch.tensor(inp), torch.tensor(out)))
            loss_fn = nn.CrossEntropyLoss()
            lrs, losses_lr = find_learning_rate(model, data, loss_fn)
            plt.figure()
            plt.plot(lrs, losses_lr)
            plt.xscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.savefig("lr_finder.png")
            print("Saved learning rate finder plot as 'lr_finder.png'. Returning to main menu.\n")

        # Handle invalid menu option
        else:
            print("Invalid input. Please choose 1, 2, 3, 4, or 5.\n")