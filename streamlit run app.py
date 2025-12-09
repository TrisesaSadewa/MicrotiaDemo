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

        /* MAIN BACKGROUND */
        .stApp {
            background-color: #f3f4f6; /* Fallback color */
            background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* OVERLAY to darken background */
        .stApp > header { background: transparent; }
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(243, 244, 246, 0.85); /* Light overlay for readability */
            pointer-events: none;
            z-index: 0;
        }

        /* CARD STYLING (General) */
        div.stContainer {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        /* TEXT COLORS */
        h1, h2, h3, h4, .stMarkdown, p, label {
            color: #111827 !important;
        }

        /* BUTTONS */
        div.stButton > button:first-child {
            background-color: #2563EB;
            color: white !important;
            border: none;
            padding: 0.75rem 1rem;
            border-radius: 8px;
        }
        div.stButton > button:first-child:hover {
            background-color: #1d4ed8;
        }

        /* STATUS BADGES */
        .status-badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-ready { background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .status-waiting { background-color: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }
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
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_segmentation_overlay(uploaded_file):
    # CRITICAL FIX: Reset file pointer before reading again
    uploaded_file.seek(0) 
    img = Image.open(uploaded_file).convert("RGBA")
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
    st.header("🦻 Setup") # Use native header instead of image to prevent broken links
    
    st.markdown("### 1. Upload Data")
    uploaded_file = st.file_uploader("Patient Image (Lateral)", type=["jpg", "png"])
    
    st.markdown("### 2. System Status")
    if uploaded_file:
        st.markdown('<div class="status-badge status-ready">✅ Image Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-waiting">⏳ Waiting for input</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 3. Calibration")
        has_ruler = st.checkbox("Image includes ruler", value=True)
        if not has_ruler:
            st.number_input("Pixels per 10mm", value=150)
            
    st.markdown("---")
    st.info("ℹ️ **Privacy Note**: Data processed locally.")

# --- MAIN CONTENT ---

# 1. Header Section
st.title("NagataGuide Pro")
st.markdown("**Automated Microtia Reconstruction Planner**")

# 2. Main Workspace
if not uploaded_file:
    # Empty State (Showcase Style)
    with st.container():
        st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #9ca3af; border-radius: 12px; background: rgba(255,255,255,0.4);">
            <h2 style="color: #4b5563 !important; margin-bottom: 10px;">No Patient Data Uploaded</h2>
            <p style="color: #6b7280; font-size: 1.1rem;">Upload a lateral ear photograph from the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("⚡ **Fast Analysis**\n\n2D to 3D in < 45 seconds")
        with c2:
            st.info("📐 **Auto-Calibration**\n\nPixel-perfect scaling logic")
        with c3:
            st.info("🖨️ **Print Ready**\n\nDirect STL exports")

else:
    # Active State
    col_act_1, col_act_2 = st.columns([1, 4])
    with col_act_1:
        if st.button("✨ Generate Guide", type="primary", use_container_width=True):
            with st.spinner("Processing anatomy..."):
                time.sleep(2) 
                st.session_state.processed = True
    
    if st.session_state.processed:
        st.markdown("---")
        # Results Dashboard
        tab1, tab2, tab3 = st.tabs(["🔍 Surgical Preview", "🧊 3D Models", "📥 Print Kit"])
        
        with tab1:
            with st.container():
                col_preview_1, col_preview_2 = st.columns(2)
                with col_preview_1:
                    uploaded_file.seek(0) # CRITICAL FIX
                    st.image(uploaded_file, caption="Original Input", use_container_width=True)
                with col_preview_2:
                    uploaded_file.seek(0) # CRITICAL FIX
                    overlay = create_segmentation_overlay(uploaded_file)
                    st.image(overlay, caption="AI Segmentation Mask", use_container_width=True)
                
        with tab2:
            with st.container():
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
            with st.container():
                st.success("Files ready for manufacturing.")
                zip_data = generate_mock_zip()
                timestamp = datetime.now().strftime("%Y%m%d")
                
                col_dl, col_list = st.columns([1, 1])
                with col_dl:
                    st.download_button(
                        label="📥 Download STL Package",
                        data=zip_data,
                        file_name=f"Nagata_{timestamp}.zip",
                        mime="application/zip",
                        type="primary"
                    )
                with col_list:
                    st.markdown("**Included Files:**")
                    st.code("BASE.stl\nHELIX.stl\nANTIHELIX.stl\nREADME.txt")
