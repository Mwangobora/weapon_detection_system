import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from PIL import Image
import os


st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    .detection-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_yolov8_model():
    """Load YOLOv8 model for gun and knife detection"""
    try:
        model_path = './runs/detect/Normal_Compressed/weights/best.pt'
        import torch
        with torch.serialization.safe_globals(['ultralytics.nn.tasks.DetectionModel']):
            model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {str(e)}")
        return None

@st.cache_resource
def load_yolov5_model():
    """Load YOLOv5 model for rifle and grenade detection"""
    try:
        model_path = './model_rifles_grenade/weapon_detection_model/weapon_model/weights/best.pt'
        import torch
        with torch.serialization.safe_globals(['models.yolo.DetectionModel']):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {str(e)}")
        return None

def get_weapon_type(class_name):
    """Determine which model to use based on detected class"""
    gun_knife_classes = ['gun', 'knife', 'handgun', 'pistol']
    rifle_grenade_classes = ['automatic rifle', 'rifle', 'grenade', 'grenade launcher', 'bazooka', 'smg', 'shotgun']

    class_lower = class_name.lower()

    if any(weapon in class_lower for weapon in gun_knife_classes):
        return 'yolov8'
    elif any(weapon in class_lower for weapon in rifle_grenade_classes):
        return 'yolov5'
    else:
        return 'both'  

def detect_weapons_combined(image, yolov8_model, yolov5_model, confidence_threshold=0.5):
    """Detect weapons using both models and combine results"""
    detections = []

    
    if yolov8_model is not None:
        try:
            results_v8 = yolov8_model(image)
            for result in results_v8:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    class_names = result.names

                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf >= confidence_threshold:
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = class_names[int(cls)]
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'class': class_name,
                                'model': 'YOLOv8'
                            })
        except Exception as e:
            st.warning(f"YOLOv8 detection failed: {str(e)}")

    
    if yolov5_model is not None:
        try:
            results_v5 = yolov5_model(image)
            detections_v5 = results_v5.pandas().xyxy[0]

            for _, detection in detections_v5.iterrows():
                if detection['confidence'] >= confidence_threshold:
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': detection['confidence'],
                        'class': detection['name'],
                        'model': 'YOLOv5'
                    })
        except Exception as e:
            st.warning(f"YOLOv5 detection failed: {str(e)}")

    return detections

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image"""
    colors = {
        'YOLOv8': (0, 255, 0),  
        'YOLOv5': (255, 0, 0)    
    }

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        model_name = detection['model']

        color = colors.get(model_name, (0, 255, 255))

        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        
        label = f"{class_name} ({model_name}) {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

        
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def process_image(uploaded_file, yolov8_model, yolov5_model, confidence_threshold):
    """Process uploaded image for weapon detection"""
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect weapons
    detections = detect_weapons_combined(image_cv, yolov8_model, yolov5_model, confidence_threshold)

    # Draw detections
    result_image = draw_detections(image_cv.copy(), detections)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    return result_image_rgb, detections

def process_video(uploaded_file, yolov8_model, yolov5_model, confidence_threshold):
    """Process uploaded video for weapon detection"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    # Open video
    cap = cv2.VideoCapture(temp_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frames
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    frame_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect weapons in frame
        detections = detect_weapons_combined(frame, yolov8_model, yolov5_model, confidence_threshold)
        all_detections.extend(detections)

        # Draw detections
        result_frame = draw_detections(frame.copy(), detections)

        # Write frame to output video
        out.write(result_frame)

        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Display current frame (every 10th frame to avoid too frequent updates)
        if frame_count % 10 == 0:
            frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption=f"Processing frame {frame_count}/{total_frames}")

    # Clean up
    cap.release()
    out.release()
    os.unlink(temp_video_path)

    return output_path, all_detections


def main():
    # Header
    st.markdown('<h1 class="main-header"> Weapon Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered weapon detection using YOLOv8 and YOLOv5</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Load models
    with st.spinner("Loading AI models..."):
        yolov8_model = load_yolov8_model()
        yolov5_model = load_yolov5_model()

    # Model status
    st.sidebar.subheader("Model Status")
    if yolov8_model is not None:
        st.sidebar.success("✅ YOLOv8 Model (Gun/Knife)")
    else:
        st.sidebar.error("❌ YOLOv8 Model Failed")

    if yolov5_model is not None:
        st.sidebar.success("✅ YOLOv5 Model (Rifle/Grenade)")
    else:
        st.sidebar.error("❌ YOLOv5 Model Failed")

    # Settings
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

   

    
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

    with tab1:
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to detect weapons"
        )

        if uploaded_image is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_image)
                st.image(original_image, use_column_width=True)

            with col2:
                st.subheader("Detection Results")
                if st.button("🔍 Detect Weapons", key="image_detect"):
                    with st.spinner("Analyzing image..."):
                        result_image, detections = process_image(uploaded_image, yolov8_model, yolov5_model, confidence_threshold)

                        st.image(result_image, use_column_width=True)

                        # Display detection summary
                        if detections:
                            st.markdown('<div class="detection-info">', unsafe_allow_html=True)
                            st.subheader(f"🎯 Found {len(detections)} weapon(s)")

                            for i, detection in enumerate(detections, 1):
                                st.write(f"**{i}.** {detection['class']} (Confidence: {detection['confidence']:.2f}) - Detected by {detection['model']}")

                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.success("✅ No weapons detected in the image")

    with tab2:
        st.subheader("Upload a Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect weapons"
        )

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("🔍 Detect Weapons in Video", key="video_detect"):
                with st.spinner("Processing video... This may take a while."):
                    output_path, all_detections = process_video(uploaded_video, yolov8_model, yolov5_model, confidence_threshold)

                    # Display processed video
                    st.subheader("Processed Video")
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)

                    # Display detection summary
                    if all_detections:
                        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
                        st.subheader(f"🎯 Total detections: {len(all_detections)}")

                        # Count detections by class
                        detection_counts = {}
                        for detection in all_detections:
                            class_name = detection['class']
                            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

                        for class_name, count in detection_counts.items():
                            st.write(f"**{class_name}**: {count} detections")

                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.success("✅ No weapons detected in the video")

                    # Clean up
                    os.unlink(output_path)

if __name__ == "__main__":
    main()