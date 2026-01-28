"""
Rickshaw Detection Application - End-to-End Object Detection using YOLOv8

This Streamlit application provides:
- Real-time rickshaw detection using a trained YOLOv8 model
- Image upload and live webcam detection modes
- Adjustable confidence threshold
- Visual bounding boxes and detection counts
- User-friendly interface with detailed statistics

Run the app:
    streamlit run app.py

Then open http://localhost:8501 in your browser
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best.pt"
RICKSHAW_CLASS_NAME = "Rickshaw"

# Page configuration
st.set_page_config(
    page_title="🚲 Rickshaw Detection System",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CACHING & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_yolo_model(model_path: str) -> YOLO:
    """
    Load and cache the YOLOv8 model.
    
    This function uses Streamlit's caching to avoid reloading the model
    on every app rerun, which significantly improves performance.
    
    Args:
        model_path: Path to the trained .pt model file
        
    Returns:
        Loaded YOLO model object
    """
    return YOLO(model_path)


def get_rickshaw_class_id(model: YOLO) -> Optional[int]:
    """
    Find the class ID for 'Rickshaw' in the model's class names.
    
    Args:
        model: The loaded YOLO model
        
    Returns:
        Integer class ID for Rickshaw, or None if not found
    """
    names = model.names  # dict or list of class names
    
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            if str(class_name).strip().lower() == "rickshaw":
                return int(class_id)
    elif isinstance(names, list):
        for class_id, class_name in enumerate(names):
            if str(class_name).strip().lower() == "rickshaw":
                return int(class_id)
    
    return None


# ============================================================================
# INFERENCE & VISUALIZATION
# ============================================================================

def draw_boxes_and_count(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict,
    target_class_id: Optional[int],
) -> Tuple[np.ndarray, int]:
    """
    Draw bounding boxes for detected rickshaws and count them.
    
    Args:
        image: Input image (BGR format from OpenCV)
        boxes_xyxy: Bounding box coordinates [[x1,y1,x2,y2], ...]
        confidences: Confidence scores for each detection
        class_ids: Class ID for each detection
        class_names: Dictionary mapping class ID to class name
        target_class_id: The class ID we're looking for (Rickshaw)
        
    Returns:
        Tuple of (annotated_image, rickshaw_count)
    """
    annotated_img = image.copy()
    rickshaw_count = 0
    
    # Draw each detection
    for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
        cls_id_int = int(cls_id)
        
        # Filter for rickshaws only
        if target_class_id is not None and cls_id_int != target_class_id:
            continue
        
        # Extract box coordinates
        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = class_names.get(cls_id_int, f"Class{cls_id_int}")
        conf_score = float(conf)
        
        # Fallback: filter by name if class ID matching failed
        if target_class_id is None and str(class_name).lower() != "rickshaw":
            continue
        
        rickshaw_count += 1
        
        # Draw green rectangle for rickshaw detection
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text with class name and confidence
        label_text = f"{class_name} {conf_score:.2f}"
        
        # Get text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        text_size, baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        text_x = x1
        text_y = max(y1 - 10, text_size[1] + 5)
        cv2.rectangle(
            annotated_img,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 10, text_y + baseline),
            (0, 255, 0),
            -1,  # Fill rectangle
        )
        
        # Draw text on background
        cv2.putText(
            annotated_img,
            label_text,
            (text_x + 5, text_y - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness,
            cv2.LINE_AA,
        )
    
    return annotated_img, rickshaw_count


def run_inference(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float,
) -> Tuple[np.ndarray, int]:
    """
    Run YOLO inference on a single image frame.
    
    Args:
        model: Loaded YOLO model
        image: Input image (BGR format)
        conf_threshold: Confidence threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (annotated_image, rickshaw_count)
    """
    # Run prediction
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False,
    )
    
    if not results or len(results) == 0:
        return image, 0
    
    result = results[0]
    
    # Check if boxes were detected
    if result.boxes is None or len(result.boxes) == 0:
        return image, 0
    
    # Extract detection data
    boxes = result.boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()
    
    # Normalize class names to dict format
    names = model.names
    if isinstance(names, dict):
        class_names_dict = {int(k): str(v) for k, v in names.items()}
    else:
        class_names_dict = {int(i): str(n) for i, n in enumerate(names)}
    
    # Find rickshaw class ID
    rickshaw_id = get_rickshaw_class_id(model)
    
    # Draw boxes and count rickshaws
    annotated, count = draw_boxes_and_count(
        image=image,
        boxes_xyxy=boxes_xyxy,
        confidences=confidences,
        class_ids=class_ids,
        class_names=class_names_dict,
        target_class_id=rickshaw_id,
    )
    
    return annotated, count


def process_video(
    video_path: str,
    model: YOLO,
    conf_threshold: float,
    progress_bar=None,
    status_text=None,
) -> Tuple[str, int, int, float]:
    """
    Process a video file and detect rickshaws in each frame.
    
    Args:
        video_path: Path to input video file
        model: Loaded YOLO model
        conf_threshold: Confidence threshold for detection
        progress_bar: Streamlit progress bar object
        status_text: Streamlit status text object
        
    Returns:
        Tuple of (output_video_path, total_frames, total_rickshaws, avg_conf)
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define video writer
    output_path = str(PROJECT_DIR / "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    total_rickshaws = 0
    confidence_sum = 0.0
    conf_count = 0
    
    rickshaw_id = get_rickshaw_class_id(model)
    names = model.names
    if isinstance(names, dict):
        class_names_dict = {int(k): str(v) for k, v in names.items()}
    else:
        class_names_dict = {int(i): str(n) for i, n in enumerate(names)}
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break
        
        frame_count += 1
        
        # Run inference
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                # Draw boxes and count
                annotated, rickshaw_count = draw_boxes_and_count(
                    image=frame,
                    boxes_xyxy=boxes_xyxy,
                    confidences=confidences,
                    class_ids=class_ids,
                    class_names=class_names_dict,
                    target_class_id=rickshaw_id,
                )
                
                total_rickshaws += rickshaw_count
                confidence_sum += np.sum(confidences[class_ids == rickshaw_id] if rickshaw_id is not None else confidences)
                conf_count += len(confidences)
                
                # Write frame to output video
                out.write(annotated)
            else:
                out.write(frame)
        else:
            out.write(frame)
        
        # Update progress
        if progress_bar is not None:
            progress_bar.progress(frame_count / total_frames)
        if status_text is not None:
            status_text.text(f"Processing: {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    avg_conf = confidence_sum / conf_count if conf_count > 0 else 0.0
    
    return output_path, total_frames, total_rickshaws, avg_conf

# Title and description
st.title("🚲 Rickshaw Detection System")
st.markdown(
    """
    **End-to-End Object Detection Application using YOLOv8**
    
    This application detects rickshaws in images and live video feeds.
    Upload an image or use your webcam to get real-time detection results.
    """
)

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model path input
    model_path_input = st.text_input(
        "Model Path",
        value=str(MODEL_PATH),
        help="Path to the trained YOLOv8 model (.pt file)",
    )
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Higher value = stricter detection (fewer false positives)",
    )
    
    # Input mode selection
    input_mode = st.radio(
        "Select Input Mode",
        options=["📸 Upload Image", "🎥 Live Webcam", "🎬 Video File"],
        index=0,
    )
    
    st.divider()
    
    # Display model info
    st.subheader("ℹ️ Model Information")
    st.info(
        f"**Model Path:** {model_path_input}\n\n"
        f"**Classes:** Rickshaw, Objects\n\n"
        f"**Task:** Object Detection"
    )


# ============================================================================
# MAIN APP LOGIC
# ============================================================================

# Load model with error handling
try:
    model = load_yolo_model(str(model_path_input))
    st.sidebar.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {model_path_input}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()


# ============================================================================
# MODE 1: IMAGE UPLOAD
# ============================================================================

if input_mode == "📸 Upload Image":
    st.subheader("📸 Image Upload & Detection")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Select an image file to run rickshaw detection",
    )
    
    if uploaded_file is not None:
        # Read uploaded image
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            st.error("❌ Could not read the image file. Please try another.")
            st.stop()
        
        # Run inference
        annotated_bgr, rickshaw_count = run_inference(
            model,
            image_bgr,
            confidence_threshold,
        )
        
        # Display results side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        with col2:
            st.subheader("Detection Results")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        
        # Display detection statistics
        st.divider()
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("🚲 Rickshaws Detected", rickshaw_count)
        
        with col_stat2:
            st.metric("🔍 Confidence Threshold", f"{confidence_threshold:.0%}")
        
        with col_stat3:
            st.metric("📏 Image Size", f"{image_bgr.shape[1]}x{image_bgr.shape[0]}")
    
    else:
        st.info("👆 Upload an image to start detection")


# ============================================================================
# MODE 2: LIVE WEBCAM
# ============================================================================

elif input_mode == "🎥 Live Webcam":
    st.subheader("🎥 Live Webcam Detection")
    
    # Initialize session state
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    
    # Control panel
    col_controls, col_display = st.columns([1, 3])
    
    with col_controls:
        st.subheader("Controls")
        
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            help="0 = default camera, 1+ = external cameras",
        )
        
        start_button = st.button(
            "▶️ Start Webcam",
            key="start_btn",
            disabled=st.session_state.webcam_running,
        )
        stop_button = st.button(
            "⏹️ Stop Webcam",
            key="stop_btn",
            disabled=not st.session_state.webcam_running,
        )
        
        if start_button:
            st.session_state.webcam_running = True
        
        if stop_button:
            st.session_state.webcam_running = False
        
        st.caption("💡 Keep this tab in focus for best results")
    
    with col_display:
        frame_placeholder = st.empty()
        count_placeholder = st.empty()
        
        if not st.session_state.webcam_running:
            frame_placeholder.info("👆 Click 'Start Webcam' to begin live detection")
        else:
            # Open webcam
            cap = cv2.VideoCapture(int(camera_index))
            
            if not cap.isOpened():
                st.error(f"❌ Cannot open camera {camera_index}. Try a different index.")
                st.session_state.webcam_running = False
                st.stop()
            
            # Webcam loop
            try:
                while st.session_state.webcam_running:
                    ret, frame_bgr = cap.read()
                    
                    if not ret or frame_bgr is None:
                        st.warning("⚠️ Failed to read from webcam")
                        break
                    
                    # Run inference
                    annotated_bgr, rickshaw_count = run_inference(
                        model,
                        frame_bgr,
                        confidence_threshold,
                    )
                    
                    # Display frame and count
                    frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")
                    count_placeholder.metric(
                        "🚲 Rickshaws Detected (Current Frame)",
                        rickshaw_count,
                    )
                    
                    time.sleep(0.03)  # ~30 FPS
            
            finally:
                cap.release()
                st.session_state.webcam_running = False


# ============================================================================
# MODE 3: VIDEO FILE
# ============================================================================

elif input_mode == "🎬 Video File":
    st.subheader("🎬 Video File Detection")
    
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
        help="Select a video file for rickshaw detection",
    )
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        video_path = PROJECT_DIR / "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.info(f"📹 Video uploaded: {uploaded_video.name}")
        
        # Process button
        if st.button("🔍 Start Detection", key="video_detect_btn"):
            with st.spinner("⏳ Processing video... This may take a moment"):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    output_path, total_frames, total_rickshaws, avg_conf = process_video(
                        str(video_path),
                        model,
                        confidence_threshold,
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )
                    
                    st.success("✅ Video processing completed!")
                    progress_bar.progress(1.0)
                    status_text.text(f"Completed: {total_frames}/{total_frames} frames")
                    
                    # Display results
                    st.divider()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎬 Total Frames", total_frames)
                    with col2:
                        st.metric("🚲 Total Rickshaws", total_rickshaws)
                    with col3:
                        st.metric("📊 Avg Confidence", f"{avg_conf:.2f}" if avg_conf > 0 else "N/A")
                    with col4:
                        st.metric("📈 Avg per Frame", f"{total_rickshaws/total_frames:.1f}" if total_frames > 0 else "N/A")
                    
                    st.divider()
                    st.subheader("📥 Download Detected Video")
                    
                    # Read output video
                    with open(output_path, "rb") as video_file:
                        st.download_button(
                            label="⬇️ Download Output Video",
                            data=video_file.read(),
                            file_name=f"rickshaw_detected_{uploaded_video.name}",
                            mime="video/mp4",
                        )
                    
                    # Cleanup
                    if video_path.exists():
                        video_path.unlink()
                
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    if video_path.exists():
                        video_path.unlink()
    
    else:
        st.info("👆 Upload a video file to start detection")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    ---
    **Rickshaw Detection System** 
    
    *End-to-End Object Detection Application*
    """
)
