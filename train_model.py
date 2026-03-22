import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile

# Load model
model = joblib.load("violence_model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------- ADC ----------
def adc_quantize(signal, bits=10):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    signal_norm = signal / max_val
    levels = 2 ** bits
    quantized = np.round(signal_norm * (levels / 2)) / (levels / 2)
    return quantized


# ---------- Feature Extraction ----------
def extract_features(file_path, bits=10):

    y, sr = librosa.load(file_path, sr=16000)
    y_q = adc_quantize(y, bits)

    energy = np.sum(y_q ** 2)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_q))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_q, sr=sr))

    mfcc = librosa.feature.mfcc(y=y_q, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    features = [energy, zcr, centroid]
    features.extend(mfcc_mean)

    return np.array(features)


# ---------- UI ----------
st.set_page_config(
    page_title="Violence Detection System",
    layout="centered"
)

st.title("🎙️ Audio Violence Detection System")
st.write("Upload an audio file to analyze acoustic aggression levels.")

threshold = st.slider(
    "Violence Detection Threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file:

    st.subheader("Uploaded Audio")
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract features
    features = extract_features(temp_path)
    features = scaler.transform([features])

    # Predict
    prob = model.predict_proba(features)[0][1]

    st.subheader("Analysis Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Violence Probability", f"{prob:.2f}")

    with col2:
        st.metric("Threshold", f"{threshold:.2f}")

    st.progress(float(prob))

    if prob > threshold:
        st.error("🚨 VIOLENCE DETECTED 🚨")
        st.warning("Emergency alert triggered!")
    else:
        st.success("✅ Non-Violent Audio")

    st.subheader("Probability Breakdown")

    st.write({
        "Violence": float(prob),
        "Non-Violence": float(1 - prob)
    })
