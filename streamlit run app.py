import streamlit as st
import time
import zipfile
import io
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION (16:9 Wide Layout) ---
st.set_page_config(
    page_title="NagataGuide Pro",
    page_icon="🦻",
    layout="wide",  # 16:9 Ratio
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DASHBOARD & BACKGROUND) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* MAIN BACKGROUND - Surgical Abstract */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* OVERLAY to darken background for readability */
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(17, 24, 39, 0.4); /* Dark overlay */
            pointer-events: none;
            z-index: 0;
        }

        /* TEXT VISIBILITY FIXES */
        h1, h2, h3, h4, .stMarkdown, p, label, .stCheckbox {
            color: #1f2937 !important; /* Dark text for cards */
        }
        
        /* Make text inside the sidebar white/light for contrast if sidebar is dark, 
           BUT standard streamlit sidebar is light. Let's keep cards white. */
        
        /* CARDS (Glassmorphism Containers) */
        div[data-testid="stVerticalBlock"] > div {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(10px);
        }
        
        /* SIDEBAR SPECIFIC STYLING */
        section[data-testid="stSidebar"] {
            background-color: #f9fafb; /* Light gray sidebar */
            border-right: 1px solid #e5e7eb;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
            background-color: transparent; /* Remove card effect in sidebar */
            box-shadow: none;
            padding: 0;
        }

        /* PRIMARY BUTTON */
        div.stButton > button:first-child {
            background-color: #2563EB;
            color: white !important;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.75rem 1rem;
            transition: all 0.2s;
        }
        div.stButton > button:first-child:hover {
            background-color: #1d4ed8;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transform: translateY(-1px);
        }

        /* STATUS BADGES */
        .status-badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-ready { background-color: #dcfce7; color: #166534; }
        .status-waiting { background-color: #f3f4f6; color: #4b5563; }

    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def generate_mock_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        dummy_stl = "solid mock\nendsolid mock"
        zip_file.writestr("BASE.stl", dummy_stl)
        zip_file.writestr("HELIX.stl", dummy_stl)
        zip_file.writestr("README.txt", "Print BASE first.")
    buffer.seek(0)
    return buffer

def create_mock_3d_plot():
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    c, a = 2, 1
    x = (c + a*np.cos(theta)) * np.cos(phi)
    y = (c + a*np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Blues', showscale=False)])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_segmentation_overlay(uploaded_image):
    img = Image.open(uploaded_image).convert("RGBA")
    img.thumbnail((600, 600))
    mask = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(mask)
    w, h = img.size
    draw.ellipse((w*0.3, h*0.2, w*0.7, h*0.8), fill=(37, 99, 235, 80), outline=(37, 99, 235, 255))
    return Image.alpha_composite(img, mask)

# --- STATE ---
if 'processed' not in st.session_state: st.session_state.processed = False

# --- SIDEBAR (SETUP) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=50) # Medical icon
    st.title("Setup")
    st.markdown("### 1. Upload Data")
    
    uploaded_file = st.file_uploader("Patient Image (Lateral)", type=["jpg", "png"])
    
    st.markdown("### 2. System Status")
    if uploaded_file:
        st.markdown('<span class="status-badge status-ready">✅ Image Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-waiting">⏳ Waiting for input</span>', unsafe_allow_html=True)
    
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 3. Calibration")
        has_ruler = st.checkbox("Image includes ruler", value=True)
        if not has_ruler:
            st.number_input("Pixels per 10mm", value=150)
            
    st.markdown("---")
    st.info("ℹ️ **Privacy Note**: All processing happens locally in this session. No data is stored permanently.")

# --- MAIN CONTENT ---

# 1. Header Section
col_head_1, col_head_2 = st.columns([3, 1])
with col_head_1:
    st.title("NagataGuide Pro")
    st.markdown("**Automated Microtia Reconstruction Planner**")

# 2. Main Workspace
if not uploaded_file:
    # Empty State (Showcase Style)
    st.markdown("<br>", unsafe_allow_html=True)
    container = st.container()
    with container:
        st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #cbd5e1; border-radius: 12px; background: rgba(255,255,255,0.5);">
            <h2 style="color: #64748b !important;">No Patient Data Uploaded</h2>
            <p style="color: #64748b;">Upload a lateral ear photograph from the sidebar to begin the 3D generation workflow.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Showcase
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ⚡ Fast Analysis")
            st.caption("2D to 3D in < 45 seconds")
        with c2:
            st.markdown("#### 📐 Auto-Calibration")
            st.caption("Pixel-perfect scaling logic")
        with c3:
            st.markdown("#### 🖨️ Print Ready")
            st.caption("Direct STL exports")

else:
    # Active State
    
    # Action Bar
    col_act_1, col_act_2 = st.columns([1, 4])
    with col_act_1:
        if st.button("✨ Generate Guide", type="primary", use_container_width=True):
            with st.spinner("Processing anatomy..."):
                time.sleep(2) # Fake processing
                st.session_state.processed = True
    
    if st.session_state.processed:
        # Results Dashboard
        tab1, tab2, tab3 = st.tabs(["🔍 Surgical Preview", "🧊 3D Models", "📥 Print Kit"])
        
        with tab1:
            col_preview_1, col_preview_2 = st.columns(2)
            with col_preview_1:
                st.image(uploaded_file, caption="Original Input", use_container_width=True)
            with col_preview_2:
                overlay = create_segmentation_overlay(uploaded_file)
                st.image(overlay, caption="AI Segmentation Mask", use_container_width=True)
                
        with tab2:
            st.markdown("### Interactive Geometry Inspection")
            c_3d_1, c_3d_2 = st.columns([2, 1])
            with c_3d_1:
                fig = create_mock_3d_plot()
                st.plotly_chart(fig, use_container_width=True)
            with c_3d_2:
                st.markdown("**Metrics**")
                st.metric("Base Height", "2.1 mm", "+0.1mm")
                st.metric("Cartilage Vol", "14.2 cc")
                st.metric("Est. Print Time", "4h 20m")
                
        with tab3:
            st.success("Files ready for manufacturing.")
            zip_data = generate_mock_zip()
            timestamp = datetime.now().strftime("%Y%m%d")
            
            st.download_button(
                label="📥 Download STL Package",
                data=zip_data,
                file_name=f"Nagata_{timestamp}.zip",
                mime="application/zip",
                type="primary"
            )
            
            st.markdown("### Included Files")
            st.code("BASE.stl\nHELIX.stl\nANTIHELIX.stl\nREADME.txt")
