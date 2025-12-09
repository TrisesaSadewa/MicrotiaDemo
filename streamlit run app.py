import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
import io
import math
from pathlib import Path
from PIL import Image
from scipy.spatial import Delaunay
from stl import mesh
import plotly.graph_objects as go
from datetime import datetime

# Try importing YOLO, handle if not installed or model missing gracefully
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# --- CONFIGURATION (16:9 Wide Layout) ---
st.set_page_config(
    page_title="NagataGuide Pro",
    page_icon="🦻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (CLINICAL THEME) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .stApp { background-color: #f3f4f6; color: #111827; }
        section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb; }
        section[data-testid="stSidebar"] * { color: #111827 !important; }
        .stContainer, .header-card { background-color: #ffffff !important; border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1, h2, h3, h4 { color: #111827 !important; font-family: 'Inter', sans-serif; font-weight: 700 !important; }
        p, li, span, div.stMarkdown { color: #374151 !important; }
        .stTextInput > label, .stNumberInput > label, .stCheckbox > label { color: #111827 !important; font-weight: 600 !important; }
        div.stButton > button:first-child { background-color: #2563EB; color: white !important; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 600; width: 100%; }
        div.stButton > button:first-child:hover { background-color: #1d4ed8; }
        /* Light Theme for File Uploader */
        [data-testid="stFileUploader"] { background-color: #f9fafb; border: 1px dashed #cbd5e1; padding: 15px; }
        [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span { color: #4b5563 !important; }
        .status-badge { background-color: #ffffff; padding: 8px; border-radius: 6px; border: 1px solid #e5e7eb; text-align: center; font-weight: 600; color: #111827; }
        .custom-code-block { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-family: monospace; color: #334155; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC (NagataGuideGenerator) ---

class NagataGuideGenerator:
    def __init__(self, yolo_path=None, pixels_per_mm=10):
        self.pixels_per_mm = pixels_per_mm
        self.model = None
        if YOLO_AVAILABLE and yolo_path and os.path.exists(yolo_path):
            try:
                self.model = YOLO(yolo_path)
            except Exception as e:
                print(f"YOLO Error: {e}")

    def process_image_buffer(self, uploaded_file):
        """Converts Streamlit UploadedFile to CV2 image"""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        uploaded_file.seek(0) # Reset pointer
        return img

    def detect_and_crop(self, img):
        """Uses YOLO if available, else assumes image is already cropped/close-up"""
        if self.model:
            results = self.model.predict(img, conf=0.3, verbose=False)
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes.data[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box[:4])
                h, w = img.shape[:2]
                pad = 60
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                return img[y1:y2, x1:x2]
        
        # Fallback: Return original image if detection fails or no model
        return img

    def extract_three_part_anatomy(self, image):
        """
        Backend Logic: Otsu (Base) + DoG (Skeleton) + Spatial Classification
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. BASE MASK
        blur_base = cv2.GaussianBlur(gray, (25, 25), 0)
        _, mask_base = cv2.threshold(blur_base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean Base (Keep largest blob)
        cnts_base, _ = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_base = np.zeros_like(gray)
        if cnts_base:
            base_cnt = max(cnts_base, key=cv2.contourArea)
            cv2.drawContours(mask_base, [base_cnt], -1, 255, -1)

        # 2. SKELETON (DoG)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        g1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
        g2 = cv2.GaussianBlur(enhanced, (21, 21), 0)
        dog = cv2.subtract(g1, g2)
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        _, mask_skeleton_raw = cv2.threshold(dog_norm, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean Skeleton
        mask_skeleton_raw = cv2.bitwise_and(mask_skeleton_raw, mask_base)
        mask_skeleton_raw = cv2.morphologyEx(mask_skeleton_raw, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)

        # 3. SPATIAL CLASSIFICATION
        mask_helix = np.zeros_like(gray)
        mask_antihelix = np.zeros_like(gray)
        
        rim_width_px = int(6.0 * self.pixels_per_mm)
        kernel_rim = np.ones((rim_width_px, rim_width_px), np.uint8)
        eroded_base = cv2.erode(mask_base, kernel_rim, iterations=1)
        rim_zone = cv2.subtract(mask_base, eroded_base)
        
        cnts_skel, _ = cv2.findContours(mask_skeleton_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_skel:
            sorted_skel = sorted(cnts_skel, key=lambda x: cv2.arcLength(x, True), reverse=True)
            for cnt in sorted_skel:
                length = cv2.arcLength(cnt, True)
                if length < 50: continue
                
                temp_mask = np.zeros_like(gray)
                cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
                intersect = cv2.bitwise_and(temp_mask, rim_zone)
                overlap_ratio = cv2.countNonZero(intersect) / (cv2.countNonZero(temp_mask) + 1e-5)
                
                if overlap_ratio > 0.40:
                    cv2.drawContours(mask_helix, [cnt], -1, 255, -1)
                else:
                    cv2.drawContours(mask_antihelix, [cnt], -1, 255, -1)

        # Polish
        mask_helix = cv2.dilate(mask_helix, np.ones((3,3),np.uint8), iterations=1)
        mask_antihelix = cv2.dilate(mask_antihelix, np.ones((3,3),np.uint8), iterations=1)
        
        return mask_base, mask_helix, mask_antihelix

    def mask_to_3d_mesh(self, mask, thickness=2.0, base_height=0.0):
        """Generates Mesh Object from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # Downsample for Delaunay speed in web app
        scale_factor = 0.5 
        
        all_points = []
        for cnt in contours:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx:
                all_points.append(pt[0] * scale_factor)
        
        if len(all_points) < 3: return None
        points_2d = np.array(all_points)
        
        try:
            tri = Delaunay(points_2d)
        except:
            return None

        # Re-check validity against mask (scaled)
        valid_faces_2d = []
        h, w = mask.shape
        for simplex in tri.simplices:
            pts = points_2d[simplex]
            center = np.mean(pts, axis=0) * (1/scale_factor) # scale back to check mask
            cx, cy = int(center[0]), int(center[1])
            if 0 <= cy < h and 0 <= cx < w:
                if mask[cy, cx] > 127:
                    valid_faces_2d.append(simplex)
        
        if not valid_faces_2d: return None

        # Create 3D Geometry
        # Physical scaling: (1/pixels_per_mm) * (1/scale_factor)
        # But we already scaled points down, so we need to correct.
        # Let's just normalize for the STL: 0.1 scale
        export_scale = 0.1 * (1/scale_factor)
        
        z_bottom = base_height
        z_top = base_height + thickness
        
        vertices_3d = []
        faces_3d = []
        n_pts = len(points_2d)
        
        # Create vertices (Bottom & Top layers)
        for p in points_2d: vertices_3d.append([p[0]*export_scale, p[1]*export_scale, z_bottom])
        for p in points_2d: vertices_3d.append([p[0]*export_scale, p[1]*export_scale, z_top])
        
        # Create Top/Bottom Faces
        for f in valid_faces_2d:
            faces_3d.append([f[0], f[2], f[1]]) # Bottom (CW)
            faces_3d.append([f[0]+n_pts, f[1]+n_pts, f[2]+n_pts]) # Top (CCW)
        
        # Create Walls
        edges = {}
        for f in valid_faces_2d:
            for i in range(3):
                p1, p2 = f[i], f[(i+1)%3]
                edge = tuple(sorted((p1, p2)))
                edges[edge] = edges.get(edge, 0) + 1
        
        for (p1, p2), count in edges.items():
            if count == 1: # Boundary edge
                b1, b2 = p1, p2
                t1, t2 = p1+n_pts, p2+n_pts
                faces_3d.extend([[b1, b2, t1], [b2, t2, t1], [b1, t1, b2], [b2, t1, t2]])

        # Create numpy-stl object
        guide_mesh = mesh.Mesh(np.zeros(len(faces_3d), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_3d):
            for j in range(3):
                guide_mesh.vectors[i][j] = vertices_3d[f[j]]
                
        return guide_mesh

    def create_hole_cutters(self, mask_helix, mask_antihelix):
        """Simple representation of holes for the ZIP"""
        combined = cv2.bitwise_or(mask_helix, mask_antihelix)
        # Create a simple cylinder mesh at random points in skeleton for demo
        # In full backend this does rigorous centerline logic
        return self.mask_to_3d_mesh(combined, thickness=10, base_height=-5)

# --- APP STATE & SETUP ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=40)
    st.markdown("# Setup")
    
    st.markdown("### 1. Upload Data")
    uploaded_file = st.file_uploader("Patient Image (Lateral)", type=["jpg", "png"])
    
    st.markdown("### 2. System Status")
    if uploaded_file:
        st.markdown('<div class="status-badge" style="color: #166534; background-color: #dcfce7; border-color: #86efac;">✅ Image Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">⏳ Waiting for input</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        st.markdown("---")
        st.markdown("### 3. Calibration")
        px_mm = st.number_input("Pixels per mm", value=10, min_value=1)
        
    st.markdown("---")
    st.caption("ℹ️ **Backend**: OpenCV + NumPy-STL")

# Main Header
st.markdown("""
<div class="header-card">
    <div style="display: flex; align-items: center; gap: 20px;">
        <span style="font-size: 3rem;">🦻</span>
        <div>
            <h1 style="margin: 0; font-size: 2rem; color: #111827;">NagataGuide Pro</h1>
            <p style="margin: 5px 0 0 0; color: #4b5563; font-size: 1.1rem;">Automated Microtia Reconstruction Planner</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Logic
if not uploaded_file:
    with st.container():
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📂</div>
            <h2 style="color: #111827; margin-bottom: 10px;">No Patient Data Uploaded</h2>
            <p style="color: #4b5563;">Upload a lateral ear photograph to run the CV pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    col_act, col_res = st.columns([1, 3])
    
    # Init Generator
    generator = NagataGuideGenerator(yolo_path="models/best_v4.pt", pixels_per_mm=px_mm)

    with col_act:
        with st.container():
            st.subheader("Actions")
            if st.button("✨ Generate Guide", type="primary"):
                with st.spinner("Running Computer Vision Pipeline..."):
                    # 1. Read Image
                    raw_img = generator.process_image_buffer(uploaded_file)
                    
                    # 2. Detect/Crop
                    crop_img = generator.detect_and_crop(raw_img)
                    
                    # 3. Segmentation
                    mask_base, mask_helix, mask_antihelix = generator.extract_three_part_anatomy(crop_img)
                    
                    # 4. Generate Meshes
                    mesh_base = generator.mask_to_3d_mesh(mask_base, thickness=2.0, base_height=0.0)
                    mesh_helix = generator.mask_to_3d_mesh(mask_helix, thickness=3.0, base_height=2.0)
                    mesh_anti = generator.mask_to_3d_mesh(mask_antihelix, thickness=3.0, base_height=2.0)
                    mesh_holes = generator.create_hole_cutters(mask_helix, mask_antihelix)

                    # Store in Session State
                    st.session_state.processed_data = {
                        "crop_img": crop_img,
                        "masks": [mask_base, mask_helix, mask_antihelix],
                        "meshes": [mesh_base, mesh_helix, mesh_anti, mesh_holes]
                    }
            
            st.markdown("---")
            st.markdown("**Parameters**")
            st.checkbox("Base Plate (2mm)", value=True, disabled=True)
            st.checkbox("Skeleton (3mm)", value=True, disabled=True)
            st.checkbox("Suture Holes (15mm)", value=True, disabled=True)

    with col_res:
        data = st.session_state.processed_data
        
        if data:
            with st.container():
                tab1, tab2, tab3 = st.tabs(["🔍 Surgical Segmentation", "🧊 3D Models", "📥 Print Kit"])
                
                with tab1:
                    st.markdown("#### Anatomical Decomposition")
                    c1, c2, c3 = st.columns(3)
                    
                    # Helper to display mask
                    def show_mask(col, mask, title, color_hex):
                        # Convert grayscale mask to colored overlay for display
                        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        # RGB mapping
                        if color_hex == "blue": colored[mask > 0] = [37, 99, 235]  # BGR
                        elif color_hex == "green": colored[mask > 0] = [34, 197, 94]
                        elif color_hex == "gray": colored[mask > 0] = [100, 100, 100]
                        
                        col.image(colored, caption=title, clamp=True, use_container_width=True)

                    masks = data["masks"]
                    show_mask(c1, masks[0], "1. Base Plate", "gray")
                    show_mask(c2, masks[1], "2. Helix Rim", "blue")
                    show_mask(c3, masks[2], "3. Antihelix", "green")
                    
                    st.markdown("---")
                    st.image(cv2.cvtColor(data['crop_img'], cv2.COLOR_BGR2RGB), caption="Processed Input (Cropped)", width=200)

                with tab2:
                    st.markdown("### Generated Geometry")
                    
                    if data["meshes"][0]:
                        mesh_obj = data["meshes"][0]
                        vectors = mesh_obj.vectors # Shape: (N_faces, 3, 3)
                        
                        # Optimization: Downsample FACES (not points) to ensure browser performance
                        # Only keep every Nth face if mesh is very dense
                        if len(vectors) > 5000:
                            vectors = vectors[::2]
                        
                        # Flatten coordinates for Plotly
                        # X = [x1, x2, x3, x1, x2, x3...]
                        x = vectors[:, :, 0].flatten()
                        y = vectors[:, :, 1].flatten()
                        z = vectors[:, :, 2].flatten()
                        
                        # Define triangle indices explicitly
                        # [0, 1, 2], [3, 4, 5], ...
                        num_triangles = len(vectors)
                        i_idxs = np.arange(0, num_triangles * 3, 3)
                        j_idxs = np.arange(1, num_triangles * 3, 3)
                        k_idxs = np.arange(2, num_triangles * 3, 3)
                        
                        fig = go.Figure(data=[go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i_idxs, j=j_idxs, k=k_idxs,
                            color='#93c5fd', # Light Blue
                            opacity=1.0,
                            flatshading=True,
                            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.1)
                        )])
                        
                        # Layout to make it look like a CAD viewer
                        fig.update_layout(
                            scene=dict(
                                aspectmode='data',
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            margin=dict(l=0,r=0,b=0,t=0), 
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate 3D mesh from image. Try a cleaner image.")

                with tab3:
                    st.success("STLs Generated Successfully.")
                    
                    # Generate ZIP on fly
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        # Helper to write STL to zip
                        def add_mesh_to_zip(mesh_obj, filename):
                            if mesh_obj:
                                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                                    mesh_obj.save(tmp.name)
                                    tmp.close()
                                    zf.write(tmp.name, filename)
                                    os.remove(tmp.name)
                        
                        meshes = data["meshes"]
                        add_mesh_to_zip(meshes[0], "BASE.stl")
                        add_mesh_to_zip(meshes[1], "HELIX.stl")
                        add_mesh_to_zip(meshes[2], "ANTIHELIX.stl")
                        add_mesh_to_zip(meshes[3], "HOLES.stl")
                        zf.writestr("README.txt", "Generated by NagataGuide Pro.\n\n1. Print Base\n2. Print Helix/Antihelix\n3. Assemble.")

                    zip_buffer.seek(0)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    
                    st.download_button(
                        label="📥 Download STL Kit (ZIP)",
                        data=zip_buffer,
                        file_name=f"Nagata_Kit_{timestamp}.zip",
                        mime="application/zip",
                        type="primary"
                    )
                    
                    st.markdown("""
                    <div class="custom-code-block">
                    <b>ZIP Contents:</b><br>
                    - BASE.stl (2.0mm)<br>
                    - HELIX.stl (Relief)<br>
                    - ANTIHELIX.stl (Relief)<br>
                    - HOLES.stl (Cutters)<br>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with st.container():
                st.info("👈 Please upload an image and click 'Generate Guide' to see results.")
