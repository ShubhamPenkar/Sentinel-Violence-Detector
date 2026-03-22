import streamlit as st
import numpy as np
import librosa
import tempfile
import plotly.graph_objects as go
import io
import soundfile as sf
import torch

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel — Audio Violence Detection",
    page_icon="🛡️",
    layout="centered",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
  }

  .stApp {
    background: #0e1117;
  }

  /* ── Top header ── */
  .app-header {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
  }

  .app-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 0.5rem;
  }

  .app-title span {
    background: linear-gradient(90deg, #f97316, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
  }

  .app-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 1px;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(249,115,22,0.08);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #fb923c;
    margin-top: 1.2rem;
  }

  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e293b;
    gap: 0;
  }

  .stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    background: transparent;
    border: none;
    padding: 0.7rem 1.4rem;
    letter-spacing: 0.3px;
  }

  .stTabs [aria-selected="true"] {
    color: #fb923c !important;
    border-bottom: 2px solid #f97316 !important;
    background: transparent !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #131720;
    border: 1px dashed #1e293b;
    border-radius: 8px;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    background: linear-gradient(135deg, #f97316, #ef4444) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important;
    transition: opacity 0.2s ease !important;
  }

  .stButton > button:hover {
    opacity: 0.88 !important;
  }

  /* ── Section label ── */
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #334155;
    margin: 2rem 0 0.75rem;
  }

  /* ── Metric strip ── */
  .metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1.2rem 0;
  }

  .metric-tile {
    background: #131720;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 0.8rem;
    text-align: center;
  }

  .metric-tile .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 500;
    color: #fb923c;
    line-height: 1;
  }

  .metric-tile .lbl {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #334155;
    margin-top: 5px;
  }

  /* ── Result card ── */
  @keyframes fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-wrap {
    animation: fade-up 0.35s ease forwards;
    margin: 1.2rem 0;
  }

  .result-card {
    border-radius: 12px;
    padding: 2rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
  }

  .result-icon {
    font-size: 2.2rem;
    line-height: 1;
    flex-shrink: 0;
  }

  .result-text-group { flex: 1; }

  .result-label {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 0.3rem;
  }

  .result-sublabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    opacity: 0.6;
  }

  .result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    flex-shrink: 0;
  }

  .card-violent {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.35);
  }
  .card-violent .result-label,
  .card-violent .result-confidence { color: #f87171; }
  .card-violent .result-sublabel   { color: #f87171; }

  .card-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(16,185,129,0.05));
    border: 1px solid rgba(34,197,94,0.3);
  }
  .card-safe .result-label,
  .card-safe .result-confidence { color: #4ade80; }
  .card-safe .result-sublabel   { color: #4ade80; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #334155 !important;
    letter-spacing: 2px !important;
    background: transparent !important;
    border: none !important;
    border-top: 1px solid #1e293b !important;
  }

  .streamlit-expanderContent {
    background: #131720 !important;
    border: 1px solid #1e293b !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
  }

  div[data-testid="stAudio"] {
    background: #131720;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.5rem;
  }

  .stSpinner > div { border-top-color: #f97316 !important; }

  .prob-row {
    display: flex;
    gap: 10px;
    margin-top: 10px;
  }

  .prob-item {
    flex: 1;
    text-align: center;
    padding: 0.55rem 1rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
  }

  .prob-safe {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
  }

  .prob-violent {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    color: #f87171;
  }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    MODEL_DIR = "wav2vec2_model"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(
                    MODEL_DIR, use_safetensors=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    return processor, model, device

try:
    processor, model, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)


# ─── Constants ────────────────────────────────────────────────────────────────
SR             = 16000
MAX_LEN        = SR * 4
THRESHOLD      = 0.65
BEAT_THRESHOLD = 1.8
LOUD_RATIO_MIN = 0.20


# ─── Audio Helpers ────────────────────────────────────────────────────────────
def preprocess_audio(file_path):
    y, _ = librosa.load(file_path, sr=SR)
    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    else:
        y = y[:MAX_LEN]
    return y

def music_check(y):
    """
    Only block if strong regular beat AND calm audio.
    Violence with background music still passes through to wav2vec2.
    """
    try:
        _, beats      = librosa.beat.beat_track(y=y, sr=SR)
        beat_strength = len(beats) / (len(y) / SR)

        # Aggression indicators — if audio is loud/chaotic let it through
        rms = np.sqrt(np.mean(y**2))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        is_music = beat_strength > BEAT_THRESHOLD
        is_calm  = rms < 0.15 and zcr < 0.08   # stricter — only very calm quiet music gets blocked

        return is_music and is_calm
    except:
        return False

def energy_gate(y):
    rms_frames = librosa.feature.rms(y=y)[0]
    loud_ratio  = np.mean(rms_frames > np.percentile(rms_frames, 70))
    return loud_ratio >= LOUD_RATIO_MIN

def predict(y):
    # Always run wav2vec2 first to get real probabilities
    inputs       = processor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Gate 1: music check — calm quiet music only
    if music_check(y):
        return 0, probs, "MUSIC_DETECTED"

    # Gate 2: threshold + energy gate
    if probs[1] >= THRESHOLD and energy_gate(y):
        return 1, probs, "THRESHOLD_AND_ENERGY"
    reason = "BELOW_THRESHOLD" if probs[1] < THRESHOLD else "ENERGY_INSUFFICIENT"
    return 0, probs, reason

def numpy_to_wav_bytes(audio_np, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

def clean_mic_audio(y, sr=16000):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 150 / (sr / 2), btype="high")
    y    = filtfilt(b, a, y).astype(np.float32)
    n_fft = 512; hop = 128
    nf    = int(0.3 * sr / hop)
    D     = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, ph = np.abs(D), np.angle(D)
    noise_profile = np.mean(mag[:, :nf], axis=1, keepdims=True)
    mag_clean     = np.maximum(mag - noise_profile * 2.0, 0.0)
    y_clean = librosa.istft(mag_clean * np.exp(1j * ph), hop_length=hop, length=len(y))
    y_trim, _ = librosa.effects.trim(y_clean, top_db=20)
    if len(y_trim) < sr * 0.3: y_trim = y_clean
    peak = np.max(np.abs(y_trim))
    if peak > 0: y_trim = y_trim / peak * 0.9
    return y_trim


# ─── Plot Helpers ─────────────────────────────────────────────────────────────
def plot_waveform(y):
    times = np.linspace(0, len(y) / SR, num=len(y))
    step  = max(1, len(y) // 2000)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=times[::step], y=y[::step], mode="lines",
        line=dict(color="#f97316", width=1.2),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)"
    ))
    fig.update_layout(
        height=130,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#131720",
        margin=dict(l=0, r=0, t=8, b=8),
        xaxis=dict(title="", gridcolor="#1e293b",
                   tickfont=dict(size=9, color="#334155", family="JetBrains Mono"),
                   showgrid=True, zeroline=False),
        yaxis=dict(title="", gridcolor="#1e293b",
                   tickfont=dict(size=9, color="#334155", family="JetBrains Mono"),
                   showgrid=True, zeroline=False),
        showlegend=False,
        font=dict(family="JetBrains Mono")
    )
    return fig




# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🛡️ <span>Sentinel</span></div>
  <div class="app-tagline">Audio Violence Detection System — wav2vec2 Neural Engine</div>
  <div class="status-pill">
    <span class="status-dot"></span>
    MODEL ONLINE &nbsp;·&nbsp; RAVDESS + CREMA-D &nbsp;·&nbsp; ACC 88.1% &nbsp;·&nbsp; AUC 0.950
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load model: {model_error}\n\nEnsure `wav2vec2_model/` folder is in the same directory as app.py.")
    st.stop()

# ─── Input ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Upload File", "Record from Mic"])

audio_bytes  = None
audio_suffix = ".wav"

with tab1:
    uploaded = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")
    if uploaded:
        audio_bytes  = uploaded.read()
        audio_suffix = "." + uploaded.name.split(".")[-1].lower()
        st.audio(audio_bytes, format=f"audio/{audio_suffix[1:]}")

with tab2:
    if not SOUNDDEVICE_AVAILABLE:
        st.warning("Run: pip install sounddevice soundfile")
    else:
        duration = st.slider("Duration (seconds)", 2, 15, 5)
        if st.button("Start Recording"):
            with st.spinner(f"Recording for {duration}s…"):
                recording = sd.rec(int(duration * SR), samplerate=SR,
                                   channels=1, dtype="float32")
                sd.wait()
            audio_np  = clean_mic_audio(recording.flatten(), SR)
            wav_bytes = numpy_to_wav_bytes(audio_np, SR)
            st.session_state["mic_audio"] = wav_bytes
            st.success("Recording complete — ready to analyse.")

        if "mic_audio" in st.session_state:
            audio_bytes  = st.session_state["mic_audio"]
            audio_suffix = ".wav"
            st.audio(audio_bytes, format="audio/wav")
            if st.button("Clear"):
                del st.session_state["mic_audio"]
                st.rerun()

# ─── Analysis ────────────────────────────────────────────────────────────────
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    with st.spinner("Analysing..."):
        try:
            y                        = preprocess_audio(temp_path)
            prediction, probs, reason = predict(y)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # Waveform
    st.markdown('<div class="section-label">Waveform</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_waveform(y), use_container_width=True,
                    config={"displayModeBar": False})

    # Metrics
    duration_s = len(y) / SR
    risk = "HIGH" if probs[1] > 0.7 else "MEDIUM" if probs[1] > 0.4 else "LOW"
    risk_color = "#f87171" if risk == "HIGH" else "#fb923c" if risk == "MEDIUM" else "#4ade80"
    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-tile">
        <div class="val">{duration_s:.1f}s</div>
        <div class="lbl">Duration</div>
      </div>
      <div class="metric-tile">
        <div class="val">{SR//1000}kHz</div>
        <div class="lbl">Sample Rate</div>
      </div>
      <div class="metric-tile">
        <div class="val">{probs[1]*100:.1f}%</div>
        <div class="lbl">Threat Score</div>
      </div>
      <div class="metric-tile">
        <div class="val" style="color:{risk_color}">{risk}</div>
        <div class="lbl">Risk Level</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Result card
    st.markdown('<div class="section-label">Assessment</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-card card-violent">
            <div class="result-icon">⚠️</div>
            <div class="result-text-group">
              <div class="result-label">Violent</div>
              <div class="result-sublabel">Threat level confirmed</div>
            </div>
            <div class="result-confidence">{probs[1]*100:.0f}%</div>
          </div>
          <div class="prob-row">
            <span class="prob-item prob-safe">Non-Violent &nbsp; {probs[0]*100:.1f}%</span>
            <span class="prob-item prob-violent">Violent &nbsp; {probs[1]*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-card card-safe">
            <div class="result-icon">✅</div>
            <div class="result-text-group">
              <div class="result-label">Non-Violent</div>
              <div class="result-sublabel">No threat detected</div>
            </div>
            <div class="result-confidence">{probs[0]*100:.0f}%</div>
          </div>
          <div class="prob-row">
            <span class="prob-item prob-safe">Non-Violent &nbsp; {probs[0]*100:.1f}%</span>
            <span class="prob-item prob-violent">Violent &nbsp; {probs[1]*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)



    # Diagnostics
    with st.expander("SYSTEM DIAGNOSTICS"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
**Violence probability** `{probs[1]:.4f}`
**Non-violence probability** `{probs[0]:.4f}`
**Decision reason** `{reason}`
**Raw prediction** `{prediction}`
            """)
        with col2:
            st.markdown(f"""
**Model** `Wav2Vec2`
**Device** `{str(device).upper()}`
**Threshold** `{THRESHOLD}`
**Beat threshold** `{BEAT_THRESHOLD} bps`
**Loud ratio min** `{LOUD_RATIO_MIN}`
            """)