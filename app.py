import streamlit as st
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

import io
import zipfile
from PIL import Image

# Helper: generate a simple 2D interference intensity map (two sources)
def generate_2d_field(wavelength=5.0, phase_diff_deg=0.0, size=256, separation=10.0):
    # simple model: two point sources placed horizontally, compute sum of waves
    x = np.linspace(-20, 20, size)
    y = np.linspace(-20, 20, size)
    xx, yy = np.meshgrid(x, y)
    # source positions
    s1 = np.array([-separation/2.0, 0.0])
    s2 = np.array([separation/2.0, 0.0])
    r1 = np.sqrt((xx - s1[0])**2 + (yy - s1[1])**2)
    r2 = np.sqrt((xx - s2[0])**2 + (yy - s2[1])**2)
    k = 2 * np.pi / float(wavelength)
    phase_diff = np.deg2rad(phase_diff_deg)
    f1 = np.cos(k * r1)
    f2 = np.cos(k * r2 + phase_diff)
    field = f1 + f2
    intensity = (field - field.min()) / (field.max() - field.min() + 1e-12)  # normalize 0..1
    return (intensity * 255).astype(np.uint8)

# Helper: capture N frames from a function that returns an image array
def capture_frames_from_func(frame_func, n_frames=10):
    frames = []
    for i in range(n_frames):
        frames.append(frame_func(i))
    return frames

# Helper: create a downloadable zip bytes object from list of numpy arrays (PNG)
def create_frames_zip_bytes(frames, prefix="frame"):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode='w') as zf:
        for idx, arr in enumerate(frames):
            # arr expected uint8 2D or 3D
            im = Image.fromarray(arr)
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            buf.seek(0)
            zf.writestr(f"{prefix}_{idx:03d}.png", buf.read())
    bio.seek(0)
    return bio.read()

# Hide Streamlit's default toolbar, header, and footer
st.markdown("""
    <style>
    [data-testid="stToolbar"], header, footer {
        visibility: hidden;
        height: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for white background and black text
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    [data-testid="stSidebar"] {
        background-color: white;
        color: black;
    }
    [data-testid="stSidebar"] * {
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    .footer {
        margin-top: 100px;
        padding: 30px;
        border-top: 1px solid #eaeaea;
        font-size: 14px;
        color: #777;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Responsive tweaks for mobile */
img { max-width:100% !important; height:auto !important; object-fit:contain !important; }

@media (max-width: 800px) {
  /* make logo smaller and reposition */
  .logo-container { left: 12px !important; font-size:28px !important; }
  /* reduce top navbar height/padding */
  .stRadio [role=radiogroup] { padding: 10px 12px !important; height:56px !important; }
  .stApp { padding-top: 78px !important; }
  /* tighten expander and control spacing */
  .streamlit-expanderHeader, .stExpanderHeader { padding:8px 12px !important; }
  /* smaller buttons on phones */
  .stButton>button, .stDownloadButton>button { padding:6px 10px !important; font-size:14px !important; }
  /* reduce footer padding */
  .footer { padding:12px !important; margin-top:40px !important; }
}

@media (max-width: 420px) {
  .logo-container { font-size:22px !important; left:8px !important; }
  .stRadio [role=radiogroup] { padding:8px 8px !important; height:56px !important; gap:10px; }
  .stApp { padding-top:72px !important; }
}
</style>
""", unsafe_allow_html=True)

# Make Streamlit buttons white with black text (override default dark styles)
st.markdown("""
    <style>
    .stButton>button, .stDownloadButton>button, .st-buttontype-primary>button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #e6e6e6 !important;
        box-shadow: none !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, .st-buttontype-primary>button:hover {
        background-color: #f7f7f7 !important;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        transform: translateY(1px) !important;
    }
    /* Make buttons' text not blue links */
    .stButton>button .css-1v3fvcr, .stDownloadButton>button .css-1v3fvcr {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# FORCE WHITE BUTTONS on all pages (Model Explorer, Simulation, Interference)
st.markdown("""
<style>
/* STRONG OVERRIDE: force ALL buttons and related interactive elements to white background + black text */
button, input[type="button"], input[type="submit"], input[type="reset"], .stButton>button, .stDownloadButton>button, .st-buttontype-primary>button, .st-buttontype-ghost>button, .st-buttontype-secondary>button, div.stButton > button, div.stDownloadButton > button {
  background-color: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
  box-shadow: none !important;
  padding: 6px 12px !important;
  border-radius: 6px !important;
}

/* also target Streamlit internal classes that sometimes wrap buttons */
.css-1emrehy, .css-1q8dd3e, .css-1cpxqw2, .css-1v3fvcr, .css-1v3fvcr > span {
  background-color: white !important;
  color: black !important;
}

/* make download buttons white too */
.stDownloadButton>button, div.stDownloadButton>button {
  background-color: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
}

/* hover/active states */
button:hover, .stButton>button:hover, .stDownloadButton>button:hover {
  background-color: #f7f7f7 !important;
  color: black !important;
}
button:active, .stButton>button:active, .stDownloadButton>button:active {
  transform: translateY(1px) !important;
}

/* ensure inner text and icons are black */
button * , .stButton>button * , .stDownloadButton>button * {
  color: black !important;
  fill: black !important;
}

/* ensure checkbox/radio labels for controls visually consistent */
label, .stRadio label, .stCheckbox label {
  color: black !important;
}

</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Optivion - Analog & Light-Based AI",
    page_icon="logo2.png",
    layout="wide"
)

# Sleek Optivion Logo with Visible Subtle Shine
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400&display=swap');

    .logo-container {
        position: fixed;
        top: 4px;
        left: 35px;
        z-index: 1300;
        font-size: 44px;
        font-weight: 300;
        font-family: 'Exo 2', sans-serif;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #111111 0%, #666666 50%, #111111 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: subtleShine 8s linear infinite;
        opacity: 0.95;
    }

    @keyframes subtleShine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    <div class="logo-container">
    <span>Optivion</span>
    </div>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
    <style>
    /* Fixed Top Navbar Styling (Top Right Alignment) */
    .stRadio [role=radiogroup] {
        display: flex;
        justify-content: flex-end;
        gap: 20px;
        background-color: white;
        border-bottom: 1px solid #e6e6e6;
        padding: 20px 40px;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        align-items: center;
        z-index: 1000;
    }
    .stApp {
        padding-top: 90px;
    }
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400&display=swap');

    .stRadio label {
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 400;
        font-size: 18px;
        letter-spacing: 1px;
    }
    .stRadio label, .stRadio div, .stRadio span, .stRadio p, .stRadio svg, .stRadio input {
        color: black !important;
        fill: black !important;
    }
    .stRadio label {
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .stRadio label:hover {
        color: #111 !important;
        text-decoration: underline;
    }
    .stRadio div[role="radio"][aria-checked="true"] label {
        color: black !important;
        font-weight: 700;
        border-bottom: 2px solid black;
    }
    </style>
""", unsafe_allow_html=True)

selected_page = st.radio(
    "Navigation Menu",  # non-empty label to remove warning
    ["Home", "Interference", "Simulation", "Model Explorer"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav"
)

# Track manual navigation override safely
if "page_override" in st.session_state:
    page = st.session_state.page_override
    del st.session_state.page_override
else:
    page = selected_page

# Home Page — Clean and Minimal
if page == "Home":
    st.markdown(
        """
        <div style="font-family: 'Exo 2', sans-serif; padding: 80px 40px 20px 40px;">
            <h1 style="font-size:56px; font-weight:200; color:#111; letter-spacing:3px; margin-bottom:10px;">Optivion</h1>
            <p style="font-size:22px; color:#333; letter-spacing:1.5px; margin-bottom:5px;">Analog & Light-Based Computation for AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Project Overview / About Section
    st.markdown("""
<div style="margin-top:0px; font-family: 'Exo 2', sans-serif; max-width:800px; margin-left:40px; margin-right:auto;">
    <p style="font-size:16px; color:#333; line-height:1.6;">
        Optivion is an innovative platform that combines analog signal processing and light-based computation with machine learning models.
        Explore simulations, visualize interference patterns, and experiment with ML models in real-time.
    </p>
</div>
    """, unsafe_allow_html=True)

    # Animated Card CSS for Problem → Solution → Impact and How It Works sections
    st.markdown("""
<style>
.animated-card {
    position: relative;
    background: white;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e6e6e6;
    overflow: hidden;
    transition: transform 0.3s, box-shadow 0.3s;
}
.animated-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 10px rgba(0, 50, 150, 0.15); /* Slight darker blue glow on hover */
}
.animated-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,50,150,0.05) 0%, transparent 70%);
    animation: cardMotion 6s linear infinite;
    z-index: 0;
}
.animated-card * {
    position: relative; /* keep text above animation */
    z-index: 1;
}
@keyframes cardMotion {
    0% { transform: rotate(0deg) translate(0,0); }
    50% { transform: rotate(180deg) translate(3px,3px); }
    100% { transform: rotate(360deg) translate(0,0); }
}
</style>
""", unsafe_allow_html=True)

    # Problem → Solution → Impact Section - Centered with animated-card class and white background, black text
    st.markdown("""
<div style="margin-top: 40px; font-family: 'Exo 2', sans-serif; max-width: 900px; margin-left:auto; margin-right:auto; text-align:center;">
    <h2 style="font-weight: 400; color: #111;">Problem, Solution & Impact</h2>
    <div style="display:flex; gap: 30px; margin-top: 20px; justify-content:center; cursor: pointer;">
        <div id="problem" class="animated-card" style="flex: 1;">
            <h3 style="font-weight: 600; color: #000;">Problem</h3>
            <p style="color: #000; line-height: 1.5;">
                Traditional digital computation struggles with energy efficiency and real-time analog signal processing,
                limiting advancements in AI that rely on physical phenomena like light interference.
            </p>
        </div>
        <div id="solution" class="animated-card" style="flex: 1;">
            <h3 style="font-weight: 600; color: #000;">Solution</h3>
            <p style="color: #000; line-height: 1.5;">
                Optivion leverages analog and light-based computation techniques integrated with ML models,
                enabling efficient, real-time simulations and visualizations that bridge the gap between physical systems and AI.
            </p>
        </div>
        <div id="impact" class="animated-card" style="flex: 1;">
            <h3 style="font-weight: 600; color: #000;">Impact</h3>
            <p style="color: #000; line-height: 1.5;">
                This approach promises breakthroughs in energy-efficient AI hardware, faster simulations,
                and novel machine learning paradigms inspired by physical analog processes.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # How It Works Section (updated colors and centering) with animated-card class and white background, black text
    st.markdown("""
<div style="margin-top: 40px; font-family: 'Exo 2', sans-serif; max-width: 900px; margin-left: auto; margin-right: auto; text-align: center;">
    <h2 style="font-weight: 400; color: #111;">How It Works</h2>
    <div style="display:flex; gap:20px; justify-content:center; margin-top:20px; cursor:pointer;">
        <div class="animated-card" style="flex:1;">
            <h3 style="font-weight:600; color: #000;">Interference Pattern</h3>
            <p style="color: #000; line-height:1.5;">Visualize light interference with real-time analog simulation.</p>
        </div>
        <div class="animated-card" style="flex:1;">
            <h3 style="font-weight:600; color: #000;">Analog Signal ML Model Visualization</h3>
            <p style="color: #000; line-height:1.5;">Explore machine learning models integrated with analog signal data.</p>
        </div>
        <div class="animated-card" style="flex:1;">
            <h3 style="font-weight:600; color: #000;">Real-Time Simulation</h3>
            <p style="color: #000; line-height:1.5;">Experience interactive analog simulations in real-time with adjustable parameters.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Step 1-4 Carousel with Blue Neon Glow
    st.markdown("""
<style>
.step-card {
    position: relative;
    background: white;
    border-radius: 50%;
    width: 160px;
    height: 160px;
    padding: 20px;
    border: 1px solid #e6e6e6;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    transition: transform 0.3s, box-shadow 0.5s;
    box-shadow: 0 0 20px rgba(0, 150, 255, 0.5);
}
.step-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 50px rgba(0, 150, 255, 0.8);
}
.step-card h3 {
    font-weight: 600;
    color: #111;
    font-size: 16px;
    margin: 5px 0;
}
.step-card p {
    font-size: 12px;
    color: #333;
    line-height: 1.2;
}
.glow {
    position: absolute;
    width: 120%;
    height: 120%;
    border-radius: 50%;
    top: -10%;
    left: -10%;
    background: radial-gradient(circle, rgba(0,150,255,0.25) 0%, transparent 70%);
    animation: glowPulse 2.5s infinite alternate;
    z-index: -1;
}
@keyframes glowPulse {
    0% { transform: scale(0.95); opacity: 0.7; }
    50% { transform: scale(1.05); opacity: 0.9; }
    100% { transform: scale(0.95); opacity: 0.7; }
}
</style>

<div style="display:flex; gap:30px; justify-content:center; margin-top:60px;">
    <div class="step-card">
        <div class="glow"></div>
        <h3>Step 1</h3>
        <p>Explore Signals</p>
    </div>
    <div class="step-card">
        <div class="glow"></div>
        <h3>Step 2</h3>
        <p>Simulate Interference</p>
    </div>
    <div class="step-card">
        <div class="glow"></div>
        <h3>Step 3</h3>
        <p>Explore ML Models</p>
    </div>
    <div class="step-card">
        <div class="glow"></div>
        <h3>Step 4</h3>
        <p>Experiment & Learn</p>
    </div>
</div>
""", unsafe_allow_html=True)

    # Footer Section
    st.markdown("""
        <div class="footer">
            <p>© 2025 Optivion. Built with passion and innovation.</p>
            <p>Contact: <a href="mailto:shreyasigh03@gmail.com">shreyasigh03@gmail.com</a></p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Page-level override for Interference: force controls and buttons white */
button, .stButton>button, .stDownloadButton>button, .stSelectbox, .stSelectbox select, .stExpander, details[role="group"] summary {
  background: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
}
button * { color: black !important; fill: black !important; }
.stSelectbox select, .stSelectbox option { background: white !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

if page == "Interference":
    # Interference page — controls left, canvas right, help in expanders
    controls_col, view_col = st.columns([1,2])

    # Controls (kept inside an expander so we don't remove existing UI)
    with controls_col:
        with st.expander("Controls", expanded=True):
            # Presets / Randomizer (callbacks preserved)
            if "phase_diff" not in st.session_state:
                st.session_state.phase_diff = 90
            if "wavelength" not in st.session_state:
                st.session_state.wavelength = 5.0

            def apply_preset_cb(preset_val):
                if preset_val == "Default":
                    st.session_state['phase_diff'] = 90
                    st.session_state['wavelength'] = 5.0
                elif preset_val == "Double-slit":
                    st.session_state['phase_diff'] = 0
                    st.session_state['wavelength'] = 2.5
                elif preset_val == "High-noise":
                    st.session_state['phase_diff'] = 45
                    st.session_state['wavelength'] = 6.5
                elif preset_val == "Low-coherence":
                    st.session_state['phase_diff'] = 180
                    st.session_state['wavelength'] = 9.0
                else:  # Random
                    st.session_state['phase_diff'] = int(np.random.uniform(0,360))
                    st.session_state['wavelength'] = float(np.round(np.random.uniform(1.0,10.0),2))

            def randomize_cb():
                st.session_state['phase_diff'] = int(np.random.uniform(0,360))
                st.session_state['wavelength'] = float(np.round(np.random.uniform(1.0,10.0),2))

            preset_col1, preset_col2 = st.columns([3,2])
            with preset_col1:
                preset = st.selectbox("Preset", ["Default","Double-slit","High-noise","Low-coherence","Random"])    
            with preset_col2:
                st.button("Apply Preset", on_click=apply_preset_cb, args=(preset,))

            st.button("Randomize", on_click=randomize_cb)

            st.write("")

            # Sliders
            phase_diff = st.slider(
                "Phase Difference (degrees)",
                0,
                360,
                value=st.session_state.phase_diff,
                step=1,
                key="phase_diff"
            )
            wavelength = st.slider(
                "Wavelength (arbitrary units)",
                1.0,
                10.0,
                value=st.session_state.wavelength,
                step=0.1,
                key="wavelength"
            )

            # Animation controls
            if "playing_interf" not in st.session_state:
                st.session_state.playing_interf = False
            if "speed_interf" not in st.session_state:
                st.session_state.speed_interf = 1.0
            if "loop_interf" not in st.session_state:
                st.session_state.loop_interf = False

            play_col1, play_col2 = st.columns([1,1])
            with play_col1:
                if st.button("Play 1D", key="play1d"):
                    st.session_state.playing_interf = True
            with play_col2:
                if st.button("Stop 1D", key="stop1d"):
                    st.session_state.playing_interf = False

            st.slider("Speed", 0.25, 4.0, value=st.session_state.speed_interf, step=0.25, key="speed_interf")
            st.checkbox("Loop", value=st.session_state.loop_interf, key="loop_interf")

            st.markdown("---")
            with st.expander("Help / Tips", expanded=False):
                st.write("Adjust phase and wavelength. Use Play to animate phase offset and Loop for continuous preview.")

    # Canvas and exporters
    with view_col:
        st.markdown("<h3 style='margin-top:6px;'>1D Interference (preview)</h3>", unsafe_allow_html=True)
        placeholder_1d = st.empty()

        def render_1d_frame(phase_offset_deg):
            fig, ax = plt.subplots(figsize=(8,2.5))
            x = np.linspace(0, 10, 500)
            y1 = np.sin(2 * np.pi * x / st.session_state.wavelength)
            y2 = np.sin(2 * np.pi * x / st.session_state.wavelength + np.deg2rad((st.session_state.phase_diff + phase_offset_deg) % 360))
            resultant = y1 + y2
            ax.plot(x, y1, label="Wave 1", linestyle="--", alpha=0.6)
            ax.plot(x, y2, label="Wave 2", linestyle="--", alpha=0.6)
            ax.plot(x, resultant, label="Resultant")
            ax.set_title("Light Interference Pattern (animated)")
            ax.set_xlabel("Position")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True, alpha=0.3)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert('RGB')

        # 1D animation (bounded loop, respects speed and loop settings)
        def play_1d_animation(frames=90):
            speed = max(0.25, float(st.session_state.get('speed_interf',1.0)))
            delay = max(0.01, 0.12 / speed)
            i = 0
            while st.session_state.playing_interf:
                im = render_1d_frame(i)
                placeholder_1d.image(im, width='stretch')
                time.sleep(delay)
                i = (i + 8) % 360
                if not st.session_state.loop_interf and i >= frames:
                    st.session_state.playing_interf = False
                    break

        # start or show a single frame
        if st.session_state.playing_interf:
            play_1d_animation()
        else:
            placeholder_1d.image(render_1d_frame(0), width='stretch')

        st.markdown("<h3 style='margin-top:12px;'>2D Interference Viewer</h3>", unsafe_allow_html=True)
        preview_place = st.empty()

        def gen_2d_frame(i, size, separation):
            ph = (st.session_state.phase_diff + i * 6) % 360
            arr = generate_2d_field(
                wavelength=st.session_state.wavelength,
                phase_diff_deg=ph,
                size=size,
                separation=separation
            )
            fig, ax = plt.subplots(figsize=(8,2.5))
            ax.imshow(arr, cmap='gray', aspect='auto')
            ax.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return np.array(Image.open(buf).convert('L'))

        # 2D controls (kept minimal here in view_col)
        size = st.selectbox("Resolution", [128, 256, 384], index=1, key="interf_size")
        separation = st.slider("Source separation", 2.0, 30.0, 10.0, 0.5, key="interf_sep")

        if "playing_2d" not in st.session_state:
            st.session_state.playing_2d = False

        if st.button("Play 2D", key="play2d_view"):
            st.session_state.playing_2d = True
        if st.button("Stop 2D", key="stop2d_view"):
            st.session_state.playing_2d = False

        # animate 2D preview
        if st.session_state.playing_2d:
            frames = []
            speed = max(0.25, float(st.session_state.get('speed_interf',1.0)))
            delay = max(0.01, 0.12 / speed)
            i = 0
            while st.session_state.playing_2d:
                frame = gen_2d_frame(i, size, separation)
                preview_place.image(frame, clamp=True, channels='L', width='stretch')
                frames.append(frame)
                time.sleep(delay)
                i = (i + 1) % 360
                if not st.session_state.loop_interf and len(frames) > 60:
                    st.session_state.playing_2d = False
                    break
            st.session_state._last_2d_frames = frames
        else:
            preview_place.image(gen_2d_frame(0, size, separation), width='stretch')
            st.session_state._last_2d_frames = [gen_2d_frame(0, size, separation)]

        # Export controls
        exp_col_a, exp_col_b = st.columns([1,1])
        with exp_col_a:
            n_frames = st.number_input("Export frames", min_value=1, max_value=200, value=10, step=1, key='interf_export_n')
        with exp_col_b:
            if st.button("Capture & Download PNGs", key='interf_capture'):
                frames = [gen_2d_frame(i, size, separation) for i in range(n_frames)]
                zip_bytes = create_frames_zip_bytes(frames, prefix="interference")
                st.download_button("Download frames (zip)", data=zip_bytes, file_name="interference_frames.zip", mime="application/zip")
            if st.button("Try Create GIF", key='interf_gif'):
                # attempt GIF creation from last frames
                frames = st.session_state.get('_last_2d_frames', [])
                if len(frames) < 2:
                    st.warning("Capture at least 2 frames first (play or capture).")
                else:
                    try:
                        pil_frames = [Image.fromarray(f).convert('P') for f in frames]
                        bio = io.BytesIO()
                        pil_frames[0].save(bio, format='GIF', save_all=True, append_images=pil_frames[1:], duration=80, loop=0)
                        bio.seek(0)
                        st.download_button("Download GIF", data=bio.read(), file_name="interference.gif", mime="image/gif")
                    except Exception as e:
                        st.error(f"GIF creation failed: {e}")

    # Footer for Interference page (kept intact)
    st.markdown("""
        <div class="footer">
            <p>© 2025 Optivion. Built with passion and innovation.</p>
            <p>Contact: <a href="mailto:shreyasigh03@gmail.com">shreyasigh03@gmail.com</a></p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Page-level override for Interference: force controls and buttons white */
button, .stButton>button, .stDownloadButton>button, .stSelectbox, .stSelectbox select, .stExpander, details[role="group"] summary {
  background: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
}
button * { color: black !important; fill: black !important; }
.stSelectbox select, .stSelectbox option { background: white !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

if page == "Simulation":
    # Simulation page — controls left, canvas right, help in expander
    controls_col, view_col = st.columns([1,2])

    with controls_col:
        with st.expander("Controls", expanded=True):
            # sliders for simulation
            if "freq" not in st.session_state:
                st.session_state.freq = 2.0
            if "amp" not in st.session_state:
                st.session_state.amp = 1.0
            if "noise" not in st.session_state:
                st.session_state.noise = 0.05

            freq = st.slider("Frequency (Hz)", 1.0, 10.0, value=st.session_state.freq, step=0.1, key="freq")
            amp = st.slider("Amplitude", 0.5, 5.0, value=st.session_state.amp, step=0.1, key="amp")
            noise = st.slider("Noise Level", 0.0, 0.5, value=st.session_state.noise, step=0.01, key="noise")

            # animation controls
            if "playing_sim" not in st.session_state:
                st.session_state.playing_sim = False
            if "speed_sim" not in st.session_state:
                st.session_state.speed_sim = 1.0
            if "loop_sim" not in st.session_state:
                st.session_state.loop_sim = False

            sim_play_col1, sim_play_col2 = st.columns([1,1])
            with sim_play_col1:
                if st.button("Play Signal", key="play_signal"):
                    st.session_state.playing_sim = True
            with sim_play_col2:
                if st.button("Stop Signal", key="stop_signal"):
                    st.session_state.playing_sim = False

            st.slider("Speed", 0.25, 4.0, value=st.session_state.speed_sim, step=0.25, key="speed_sim")
            st.checkbox("Loop", value=st.session_state.loop_sim, key="loop_sim")

            st.markdown("---")
            with st.expander("Help / Tips", expanded=False):
                st.write("Increase noise to test signal robustness. Use Play to animate and capture frames for export.")

    with view_col:
        st.markdown("<h1 style='margin-top:6px;'>Analog Signal Simulation</h1>", unsafe_allow_html=True)
        sim_placeholder = st.empty()

        def render_signal_frame(phase):
            t = np.linspace(0, 2, 500)
            y1 = st.session_state.amp * np.sin(2 * np.pi * st.session_state.freq * t + phase)
            noise_data = np.random.normal(0, st.session_state.noise, size=t.shape)
            sig = y1 + noise_data
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(t, sig, label="Analog Signal (sine)")
            ax.plot(t, st.session_state.amp * np.cos(2 * np.pi * st.session_state.freq * t), label="Cosine Reference", linestyle="--", alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_ylim(-6,6)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert('RGB')

        def play_sim_animation():
            speed = max(0.25, float(st.session_state.get('speed_sim',1.0)))
            delay = max(0.01, 0.12 / speed)
            i = 0
            while st.session_state.playing_sim:
                imf = render_signal_frame(i * 0.12)
                sim_placeholder.image(imf, width='stretch')
                time.sleep(delay)
                i += 1
                if not st.session_state.loop_sim and i > 200:
                    st.session_state.playing_sim = False
                    break

        if st.session_state.playing_sim:
            play_sim_animation()
        else:
            sim_placeholder.image(render_signal_frame(0), width='stretch')

        # Frame capture & export for simulation
        sim_col1, sim_col2 = st.columns([3,1])
        with sim_col1:
            sim_n = st.number_input("Simulation export frames", min_value=1, max_value=200, value=20, step=1, key='sim_export_n')
        with sim_col2:
            st.markdown("""
<style>
#sim_capture button, [key="sim_capture"] button, div.stButton > button[kind="secondary"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #e6e6e6 !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)
            if st.button("Capture Signal Frames", key='sim_capture'):
                def sim_frame_func(i):
                    phase = i * 0.12
                    t = np.linspace(0, 2, 500)
                    y1 = st.session_state.amp * np.sin(2 * np.pi * st.session_state.freq * t + phase)
                    noise_data = np.random.normal(0, st.session_state.noise, size=t.shape)
                    sig = y1 + noise_data
                    fig_tmp, ax_tmp = plt.subplots(figsize=(4,1))
                    ax_tmp.plot(t, sig, color='black')
                    ax_tmp.axis('off')
                    buf = io.BytesIO()
                    fig_tmp.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                    plt.close(fig_tmp)
                    buf.seek(0)
                    im = Image.open(buf).convert('L')
                    return np.array(im)
                frames = capture_frames_from_func(sim_frame_func, n_frames=sim_n)
                zip_bytes = create_frames_zip_bytes(frames, prefix='sim')
                st.download_button("Download simulation frames (zip)", data=zip_bytes, file_name="simulation_frames.zip", mime="application/zip")

    # Footer for Simulation page (kept intact)
    st.markdown("""
        <div class="footer">
            <p>© 2025 Optivion. Built with passion and innovation.</p>
            <p>Contact: <a href="mailto:shreyasigh03@gmail.com">shreyasigh03@gmail.com</a></p>
        </div>
    """, unsafe_allow_html=True)

if page == "Model Explorer":
    # Model Explorer — controls left, view right, help inside expander
    controls_col, view_col = st.columns([1,2])

    with controls_col:
        with st.expander("Controls", expanded=True):
            dataset_name = st.selectbox("Select Dataset", ["Moons", "Circles", "Classification"], key='me_dataset')
            model_name = st.selectbox("Select Model", ["SVM", "Logistic Regression", "KNN"], key='me_model')

            # animation controls and settings
            if "playing_model" not in st.session_state:
                st.session_state.playing_model = False
            if "speed_model" not in st.session_state:
                st.session_state.speed_model = 1.0
            if "loop_model" not in st.session_state:
                st.session_state.loop_model = False

            model_play_a, model_play_b = st.columns([1,1])
            with model_play_a:
                if st.button("Play Model", key="play_model"):
                    st.session_state.playing_model = True
            with model_play_b:
                if st.button("Stop Model", key="stop_model"):
                    st.session_state.playing_model = False

            st.slider("Speed", 0.25, 4.0, value=st.session_state.speed_model, step=0.25, key="speed_model")
            st.checkbox("Loop", value=st.session_state.loop_model, key="loop_model")

            st.markdown("---")
            with st.expander("Help / Tips", expanded=False):
                st.write("Switch datasets and models to compare boundaries. Use Play to observe sensitivity to jitter.")

    with view_col:
        # Load dataset
        if st.session_state.get('me_dataset') == 'Moons' or dataset_name == 'Moons':
            X, y = make_moons(noise=0.3, random_state=0)
        elif st.session_state.get('me_dataset') == 'Circles' or dataset_name == 'Circles':
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
        else:
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=22, n_clusters_per_class=1)

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if st.session_state.get('me_model') == 'SVM' or model_name == 'SVM':
            model = SVC(kernel="rbf", gamma=0.8, C=1.0)
        elif st.session_state.get('me_model') == 'Logistic Regression' or model_name == 'Logistic Regression':
            model = LogisticRegression()
        else:
            model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.markdown(f"<p style='font-size:16px; color:#000;'>Model Accuracy: <strong>{score*100:.2f}%</strong></p>", unsafe_allow_html=True)

        st.markdown("<p style='color:#000;'>Tip: use the Play button beside the decision boundary to animate the boundary slightly for intuition.</p>", unsafe_allow_html=True)

        # Decision Boundary Plot setup
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        model_place = st.empty()

        def render_model_frame(jitter):
            Xj = X + np.random.normal(0, jitter, size=X.shape)
            model.fit(Xj, y)
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
            ax.scatter(Xj[:, 0], Xj[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
            ax.set_title(f"{model_name} Decision Boundary (animated)")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert('RGB')

        # animation loop (bounded and responsive)
        def play_model_animation():
            speed = max(0.25, float(st.session_state.get('speed_model',1.0)))
            delay = max(0.01, 0.12 / speed)
            j = 0
            while st.session_state.playing_model:
                im_m = render_model_frame(0.03)
                model_place.image(im_m, width='stretch')
                time.sleep(delay)
                j += 1
                if not st.session_state.loop_model and j > 300:
                    st.session_state.playing_model = False
                    break

        if st.session_state.playing_model:
            play_model_animation()
        else:
            model_place.image(render_model_frame(0.0), width='stretch')

# FINAL OVERRIDE: force selectboxes, radios, buttons, dropdowns and expander controls to white background + black text
st.markdown("""
<style>
/* FINAL OVERRIDE: force selectboxes, radios, buttons, dropdowns and expander controls to white background + black text */
/* Buttons & inputs */
button, input[type="button"], input[type="submit"], input[type="reset"], .stButton>button, .stDownloadButton>button, div.stButton > button, div.stDownloadButton > button, .st-buttontype-primary>button, .st-buttontype-secondary>button, .st-buttontype-ghost>button {
  background: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
  box-shadow: none !important;
}

/* Selectbox / dropdowns */
.stSelectbox, .stSelectbox>div, .stSelectbox>div>div, .stSelectbox>div>div>div, .stSelectbox>div>div>button, .stSelectbox button {
  background: white !important;
  color: black !important;
}
.stSelectbox select, select, option, .stSelectbox .css-1v3fvcr {
  background: white !important;
  color: black !important;
}

/* Multiselect */
.stMultiSelect, .stMultiSelect > div, .stMultiSelect select {
  background: white !important;
  color: black !important;
}

/* Radio / Checkbox labels */
.stRadio label, .stCheckbox label, label, .stRadio, .stCheckbox {
  color: black !important;
}

/* Make dropdown menu items readable */
[role="listbox"] [role="option"], .css-1v3fvcr, .css-1v3fvcr * {
  background: white !important;
  color: black !important;
}

/* Expander (controls panel) specific rules */
.stExpander, .st-expander, .stExpander st-expander, .streamlit-expander, .stExpanderHeader, .stExpanderSummary, details[role="group"] summary, details summary {
  background: white !important;
  color: black !important;
}
.stExpander button, .st-expander button, details[role="group"] summary button, details summary button {
  background: white !important;
  color: black !important;
  border: 1px solid #e6e6e6 !important;
}
.stExpander button span, .st-expander button span, details summary span {
  color: black !important;
}

/* Icons and inner spans */
button * , .stButton>button * , .stDownloadButton>button * , .stSelectbox * , .stRadio * , .stCheckbox * {
  color: black !important;
  fill: black !important;
}

/* Hover/active states */
button:hover, .stButton>button:hover, .stDownloadButton>button:hover, .stSelectbox button:hover, .stExpander button:hover {
  background-color: #f7f7f7 !important;
  color: black !important;
}
button:active, .stButton>button:active, .stDownloadButton>button:active {
  transform: translateY(1px) !important;
}

/* Ensure inputs and sliders labels contrast */
input, textarea, .stSlider, .stNumberInput, .stTextInput {
  color: black !important;
}

</style>
""", unsafe_allow_html=True)
# Footer always appears immediately