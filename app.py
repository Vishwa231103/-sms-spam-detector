import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Load and Prepare Dataset ---
data = pd.read_csv("sms.tsv", encoding='latin-1', names=["label", "message"], sep='\t')
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label_num']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# --- Prediction Function ---
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prob = model.predict_proba(msg_vec)[0][1]
    label = model.predict(msg_vec)[0]
    label_text = "spam" if label == 1 else "ham"
    return [prob, label_text]

# --- Highlight Spammy Keywords ---
def highlight_spam_words(message):
    spammy_words = ["win", "free", "prize", "congratulations", "click", "offer", "urgent", "account", "claim", "verify", "gift"]
    highlighted = message
    for word in spammy_words:
        highlighted = highlighted.replace(word, f"**:red[{word}]**")
    return highlighted

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Detector", layout="wide")
st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message to check if it's **spam** or **ham**.")

# --- Input Area ---
user_input = st.text_area("✍️ Your Message", height=100)

if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        prob, label = predict_message(user_input)
        highlighted_msg = highlight_spam_words(user_input)

        st.markdown(f"### 🔍 Prediction Result: **{label.upper()}**")
        st.progress(min(int(prob * 100), 100), text=f"{prob:.2%} Spam Probability")

        st.markdown("#### 🧠 Explanation (keyword-based):")
        st.write(highlighted_msg)

        # Log history
        st.session_state.history.append((user_input, label, prob))

# --- History Log ---
if st.session_state['history']:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history, columns=["Message", "Label", "Spam Probability"])
    st.dataframe(hist_df[::-1].reset_index(drop=True), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **P.Vishwateja**")
