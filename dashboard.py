# 🧠 AscentGPT Training Dashboard (GUI Version)
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import matplotlib.image as mpimg
from datetime import datetime

# --- Config ---
LOG_DIR = "training_logs"

def load_metadata():
    # Load metadata from each run's folder
    metadata_files = []
    for subfolder in os.listdir(LOG_DIR):
        subfolder_path = os.path.join(LOG_DIR, subfolder)
        if os.path.isdir(subfolder_path):
            meta_path = os.path.join(subfolder_path, "metadata.txt")
            if os.path.exists(meta_path):
                metadata_files.append(meta_path)
    runs = []
    for meta_path in metadata_files:
        run_info = {}
        with open(meta_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    run_info[key.strip()] = value.strip()
        run_info["Folder Name"] = os.path.basename(os.path.dirname(meta_path))
        runs.append(run_info)
    if not runs:
        print("No completed training runs found. Dashboard will remain empty.")
        return pd.DataFrame(columns=["Run", "Timestamp", "Best Avg Loss", "Final Perplexity", "Embedding Dim", "Hidden Dim", "Num Transformer Blocks", "Batch Size", "Learning Rate", "Training Duration (min)", "Folder Name"])
    return pd.DataFrame(runs)

def plot_graphs(run_info):
    # Display loss and perplexity curve images side-by-side
    run_folder = os.path.join(LOG_DIR, run_info["Folder Name"])
    loss_img_path = os.path.join(run_folder, "loss_curve.png")
    perplexity_img_path = os.path.join(run_folder, "perplexity_curve.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    if os.path.exists(loss_img_path):
        img_loss = mpimg.imread(loss_img_path)
        axs[0].imshow(img_loss)
        axs[0].axis('off')
        axs[0].set_title("Loss Curve")
    else:
        axs[0].text(0.5, 0.5, "Loss curve image not found.", ha='center', va='center')
        axs[0].axis('off')

    if os.path.exists(perplexity_img_path):
        img_perplexity = mpimg.imread(perplexity_img_path)
        axs[1].imshow(img_perplexity)
        axs[1].axis('off')
        axs[1].set_title("Perplexity Curve")
    else:
        axs[1].text(0.5, 0.5, "Perplexity curve image not found.", ha='center', va='center')
        axs[1].axis('off')

    plt.tight_layout()
    return fig

def on_run_select(event):
    selection = run_listbox.curselection()
    if not selection:
        return
    idx = selection[0]
    run_info = df_sorted.iloc[idx]

    for widget in frame_plot.winfo_children():
        widget.destroy()

    fig = plot_graphs(run_info)
    plt.close('all')
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(padx=10, pady=10)
    canvas.draw()

    # Metadata table
    metadata_label = tk.Label(frame_plot, text="📊 Training Run Metadata", font=("Helvetica", 12, "bold"))
    metadata_label.pack(padx=10, pady=(10, 0))

    canvas_wrapper = tk.Frame(frame_plot)
    canvas_wrapper.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=False)

    canvas = tk.Canvas(canvas_wrapper, height=180)
    scrollbar = tk.Scrollbar(canvas_wrapper, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    for key, value in run_info.items():
        if key == "Folder Name":
            continue
        if key == "Timestamp":
            value = value.replace("_", ":")
        row = tk.Frame(scroll_frame)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=f"{key}:", width=25, anchor='w', font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(row, text=value, anchor='w', font=("Helvetica", 10)).pack(side=tk.LEFT)

    run_folder = os.path.join(LOG_DIR, run_info["Folder Name"])
    notes_label = tk.Label(frame_plot, text="📝 Notes", font=("Helvetica", 12, "bold"))
    notes_label.pack(padx=10, pady=(20,5))
    notes_path = os.path.join(run_folder, "notes.txt")
    notes_text = scrolledtext.ScrolledText(frame_plot, height=5, wrap=tk.WORD)
    notes_text.pack(padx=10, pady=(0,10))
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            notes_text.insert(tk.END, f.read())
    else:
        notes_text.insert(tk.END, "(Add your observations here...)")

    def save_notes():
        with open(notes_path, "w") as f:
            f.write(notes_text.get("1.0", tk.END))
        messagebox.showinfo("Saved", "Notes saved successfully!")

    save_button = tk.Button(frame_plot, text="💾 Save Notes", command=save_notes)
    save_button.pack(padx=10, pady=(0,10))
    notes_text.bind("<FocusOut>", lambda e: save_notes())

def refresh_dashboard():
    # Load metadata and update the dashboard UI with sidebar list and details pane
    for widget in frame_content.winfo_children():
        widget.destroy()

    global df_sorted, run_listbox, frame_plot
    df = load_metadata()
    if df.empty:
        empty_label = tk.Label(frame_content, text="🚀 No training runs yet. Let's build something incredible!", font=("Helvetica", 14))
        empty_label.pack(pady=20)
        return

    df["Run"] = df["Run"].astype(int)

    if "Timestamp" in df.columns:
        def format_ts(ts):
            try:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ts
        df["Timestamp"] = df["Timestamp"].apply(format_ts)

    df_sorted = df.sort_values(by="Timestamp", ascending=False)

    # Sidebar frame for run list
    frame_sidebar = tk.Frame(frame_content, width=250)
    frame_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    scrollbar = tk.Scrollbar(frame_sidebar)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    run_listbox = tk.Listbox(frame_sidebar, yscrollcommand=scrollbar.set, width=40, font=("Helvetica", 10))
    scrollbar.config(command=run_listbox.yview)
    run_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    run_entries = [f"Run #{row['Run']} — {row['Timestamp']}" for _, row in df_sorted.iterrows()]
    for entry in run_entries:
        run_listbox.insert(tk.END, entry)

    # Right pane for details and plots
    frame_plot = tk.Frame(frame_content)
    frame_plot.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    run_listbox.bind("<<ListboxSelect>>", on_run_select)

def create_new_note():
    # Create a global note window to add new notes
    new_note_window = tk.Toplevel(root)
    new_note_window.title("New Note")
    new_note_window.geometry("400x300")

    tk.Label(new_note_window, text="Write your note below:", font=("Helvetica", 12)).pack(pady=10)
    note_text = scrolledtext.ScrolledText(new_note_window, wrap=tk.WORD, width=40, height=10)
    note_text.pack(padx=10, pady=10)

    def save_new_note():
        text = note_text.get("1.0", tk.END).strip()
        if text:
            note_file = os.path.join(LOG_DIR, "notes_global.txt")
            with open(note_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}]\n{text}\n\n")
            messagebox.showinfo("Saved", "New note saved!")
            new_note_window.destroy()
        else:
            messagebox.showwarning("Empty", "Please write something before saving.")

    save_btn = tk.Button(new_note_window, text="💾 Save Note", command=save_new_note)
    save_btn.pack(pady=10)

# --- Setup Window ---
root = tk.Tk()
root.title("AscentGPT Dashboard")
root.geometry("1000x800")

frame_top = tk.Frame(root)
frame_top.pack(pady=10)

refresh_btn = tk.Button(frame_top, text="🔄 Refresh Dashboard", command=refresh_dashboard)
refresh_btn.pack()

note_btn = tk.Button(frame_top, text="📝 New Note", command=create_new_note)
note_btn.pack()

frame_content = tk.Frame(root)
frame_content.pack(fill=tk.BOTH, expand=True)

refresh_dashboard()

root.protocol("WM_DELETE_WINDOW", root.destroy)

root.mainloop()