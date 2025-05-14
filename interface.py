

import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Ascent Interface", layout="centered")

st.title("🚀 Ascent: Control Panel")

# Train the model
if st.button("🧠 Train the Model"):
    with st.spinner("Training..."):
        subprocess.Popen(["python3", "model.py", "1"])

# Retokenize
if st.button("🔤 Retokenize Conversations"):
    with st.spinner("Running tokenizer..."):
        subprocess.call(["python3", "build_tokenizer.py"])
    st.success("Tokenizer complete!")

# Chat with Ascent
if st.button("💬 Chat with Ascent"):
    subprocess.Popen(["python3", "model.py", "2"])

# Open dashboard folder
if st.button("📁 Open Training Logs Folder"):
    dashboard_path = os.path.abspath("training_logs")
    subprocess.Popen(["open", dashboard_path])  # Use 'xdg-open' for Linux, 'start' for Windows


# Kill training and save
if st.button("🛑 Stop Training and Save Model"):
    confirm = st.radio("Are you sure you want to stop training?", ["No", "Yes"], index=0)
    if confirm == "Yes":
        try:
            # Kill the python training process (rudimentary; best if training runs in separate PID)
            os.system("pkill -f 'model.py 1'")
            st.success("Training stopped. If model is set to handle interrupts, it should be saved.")
        except Exception as e:
            st.error(f"Error stopping training: {e}")