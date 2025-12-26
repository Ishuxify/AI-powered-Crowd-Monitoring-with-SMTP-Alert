import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from PIL import Image
import time
from datetime import datetime
import os
import tempfile
from collections import deque
from ultralytics import YOLO
import supervision as sv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# WebRTC imports
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Page config
st.set_page_config(
    page_title="Deep Vision Crowd Monitor: AI for Density Estimation and Overcrowding Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEVICE = torch.device("cpu")
TARGET_SIZE = (512, 512)
GT_DOWNSAMPLE = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== 🔐 SECURE SMTP CREDENTIALS ====================
def get_smtp_config():
    """Load SMTP credentials from Streamlit Secrets"""
    try:
        if hasattr(st, 'secrets') and 'smtp' in st.secrets:
            return {
                'smtp_server': st.secrets['smtp']['server'],
                'smtp_port': st.secrets['smtp']['port'],
                'sender_email': st.secrets['smtp']['sender_email'],
                'sender_password': st.secrets['smtp']['sender_password']
            }
    except:
        pass
    return None

SMTP_CONFIG = get_smtp_config()

# ==================== EMAIL ALERT SYSTEM ====================
class EmailAlertSystem:
    """SMTP Email Alert System with Secure Credentials"""
    
    def __init__(self, recipient_emails, enabled=True):
        if SMTP_CONFIG is None:
            self.enabled = False
            return
        
        self.smtp_server = SMTP_CONFIG['smtp_server']
        self.smtp_port = SMTP_CONFIG['smtp_port']
        self.sender_email = SMTP_CONFIG['sender_email']
        self.sender_password = SMTP_CONFIG['sender_password']
        self.recipient_emails = recipient_emails
        self.enabled = enabled and SMTP_CONFIG is not None
        self.last_alert_time = None
        self.alert_cooldown = 300
    
    def can_send_alert(self):
        if self.last_alert_time is None:
            return True
        return (time.time() - self.last_alert_time) >= self.alert_cooldown
    
    def send_alert_email(self, subject, count, threshold, frame_info=None, image_path=None):
        if not self.enabled or not self.can_send_alert():
            return False
        
        if not self.sender_email or not self.sender_password:
            return False
        
        try:
            msg = MIMEMultipart('related')
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = subject
            
            html_body = f"""
            <html>
              <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 10px;">
                  <h2>🚨 CROWD ALERT TRIGGERED!</h2>
                </div>
                
                <div style="padding: 20px; background-color: #f5f5f5; margin-top: 20px; border-radius: 10px;">
                  <h3>Alert Details:</h3>
                  <ul style="font-size: 16px; line-height: 1.8;">
                    <li><strong>Current Count:</strong> <span style="color: #ff4444; font-size: 20px;">{int(count)}</span> people</li>
                    <li><strong>Threshold:</strong> {int(threshold)} people</li>
                    <li><strong>Status:</strong> <span style="color: #ff4444;">⚠️ EXCEEDED</span></li>
                    <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    {f'<li><strong>Frame:</strong> {frame_info}</li>' if frame_info else ''}
                  </ul>
                </div>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    image.add_header('Content-ID', '<alert_image>')
                    msg.attach(image)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.last_alert_time = time.time()
            return True
            
        except Exception as e:
            st.error(f"❌ Email failed: {e}")
            return False


# ==================== MODEL LOADING ====================
@st.cache_resource
def create_csrnet():
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    features = list(vgg.features.children())
    frontend = nn.Sequential(*features[0:23])
    backend = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, kernel_size=1, padding=0),
    )
    model = nn.Sequential(frontend, backend)
    return model

@st.cache_resource
def load_trained_model(model_path):
    model = create_csrnet()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    torch.set_num_threads(4)
    return model

@st.cache_resource
def load_yolo_model():
    with st.spinner("Loading YOLOv8 model..."):
        yolo = YOLO('yolov8n.pt')
    return yolo


# ==================== ADAPTIVE HYBRID COUNTER ====================
class AdaptiveHybridCounter:
    def __init__(self, csrnet_model, yolo_model):
        self.csrnet = csrnet_model
        self.yolo = yolo_model
        self.device = DEVICE
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30)
        self.count_history = deque(maxlen=5)
        self.box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.from_rgb_tuple((0, 255, 255)))
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=sv.Color.from_rgb_tuple((255, 255, 255)))
    
    def preprocess_for_csrnet(self, roi):
        h, w = roi.shape[:2]
        new_h = ((h // 8) * 8) if h > 8 else 8
        new_w = ((w // 8) * 8) if w > 8 else 8
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_normalized - self.mean) / self.std
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device, dtype=torch.float32), (h, w)
    
    def predict_density_roi(self, roi):
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return np.zeros((roi.shape[0], roi.shape[1])), 0.0
        original_h, original_w = roi.shape[:2]
        with torch.no_grad():
            img_tensor, _ = self.preprocess_for_csrnet(roi)
            density_map = self.csrnet(img_tensor)
            density_np = density_map.squeeze().cpu().numpy()
            density_resized = cv2.resize(density_np, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            roi_area = original_h * original_w
            full_area = 512 * 512
            scale_factor = roi_area / full_area
            raw_count = float(density_resized.sum())
            scaled_count = raw_count * scale_factor * 0.5
            scaled_count = min(scaled_count, 3.0)
        return density_resized, scaled_count
    
    def predict_density_full(self, frame):
        target_size = (512, 512)
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_normalized - self.mean) / self.std
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device, dtype=torch.float32)
            density_map = self.csrnet(img_tensor)
            density_np = density_map.squeeze().cpu().numpy()
            count = float(density_np.sum())
        return density_np, count
    
    def detect_heads_yolo(self, frame, confidence_threshold=0.4):
        results = self.yolo(frame, conf=confidence_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        return detections
    
    def create_confidence_mask(self, box_h, box_w):
        mask = np.zeros((box_h, box_w), dtype=np.float32)
        y_center, x_center = box_h // 2, box_w // 2
        for y in range(box_h):
            for x in range(box_w):
                dy = (y - y_center) / (box_h / 2) if box_h > 1 else 0
                dx = (x - x_center) / (box_w / 2) if box_w > 1 else 0
                dist = np.sqrt(dx**2 + dy**2)
                weight = max(0.5, 1.0 - (dist * 0.5))
                mask[y, x] = weight
        return mask
    
    def predict_adaptive(self, frame, yolo_conf=0.4, use_adaptive=True, density_threshold=30):
        h, w = frame.shape[:2]
        detections = self.detect_heads_yolo(frame, confidence_threshold=yolo_conf)
        yolo_count = len(detections)
        if use_adaptive and yolo_count > density_threshold:
            mode = "CSRNet Only"
            density_map_full, final_count = self.predict_density_full(frame)
            density_map_visual = cv2.resize(density_map_full, (w, h), interpolation=cv2.INTER_CUBIC)
            tracked_detections = sv.Detections.empty()
        else:
            mode = "YOLO + CSRNet"
            tracked_detections = self.tracker.update_with_detections(detections)
            density_map_visual = np.zeros((h, w), dtype=np.float32)
            final_count = 0.0
            if len(tracked_detections) > 0:
                for box in tracked_detections.xyxy:
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    roi = frame[y1:y2, x1:x2]
                    density_roi, count_roi = self.predict_density_roi(roi)
                    conf_mask = self.create_confidence_mask(density_roi.shape[0], density_roi.shape[1])
                    density_roi_weighted = density_roi * conf_mask
                    density_map_visual[y1:y2, x1:x2] += density_roi_weighted
                    final_count += min(1.0, count_roi)
        self.count_history.append(final_count)
        smoothed_count = np.mean(self.count_history)
        annotated = self.create_annotated_frame(frame, tracked_detections, density_map_visual, smoothed_count, mode, yolo_count)
        return annotated, density_map_visual, smoothed_count, mode, yolo_count
    
    def create_annotated_frame(self, frame, detections, density_map, count, mode, yolo_count):
        annotated = frame.copy()
        if density_map.max() > 0:
            density_normalized = density_map / density_map.max()
        else:
            density_normalized = density_map
        heatmap = cm.jet(density_normalized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        annotated = cv2.addWeighted(annotated, 0.6, heatmap, 0.4, 0)
        if len(detections) > 0:
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
            if detections.tracker_id is not None:
                labels = [f"ID:{tid}" for tid in detections.tracker_id]
            else:
                labels = [f"#{i+1}" for i in range(len(detections))]
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        return annotated
    
    def reset_tracker(self):
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30)
        self.count_history.clear()


# ==================== WEBRTC VIDEO PROCESSOR ====================
class CrowdVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = None
        self.yolo_conf = 0.4
        self.use_adaptive = True
        self.density_threshold = 30
        self.alert_threshold = 50
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.counter is None:
            # Just return original frame if models not loaded
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            # Process frame
            annotated, _, count, mode, yolo_count = self.counter.predict_adaptive(
                img, 
                yolo_conf=self.yolo_conf,
                use_adaptive=self.use_adaptive,
                density_threshold=self.density_threshold
            )
            
            # Add info overlay
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 120), (0, 0, 0), -1)
            mode_color = (255, 165, 0) if mode == "YOLO + CSRNet" else (147, 112, 219)
            
            cv2.putText(annotated, "Live WebRTC Detection", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Mode: {mode}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            cv2.putText(annotated, f"Count: {int(count)} | YOLO: {yolo_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if count > self.alert_threshold:
                cv2.putText(annotated, f"ALERT! Count > {self.alert_threshold}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==================== MAIN UI ====================
def main():
    st.title("🎥 Deep Vision Crowd Monitor: AI-Powered Live Detection")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("CSRNet Model Path", value="best_crowd_counter_objects.pth")
        
        st.divider()
        st.markdown("### 🎯 Detection Settings")
        alert_threshold = st.slider("Alert Threshold", 5, 100, 50, 5)
        yolo_conf = st.slider("YOLO Confidence", 0.2, 0.8, 0.4, 0.05)
        density_threshold = st.slider("Dense Crowd Threshold", 10, 100, 30, 5)
        use_adaptive = st.checkbox("Enable Adaptive Mode", value=True)
        
        st.divider()
        st.info("💡 **WebRTC Live Camera**\n\nWorks on cloud!\nBrowser will ask camera permission.")
    
    # Load models
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if not st.session_state.models_loaded:
        if Path(model_path).exists():
            with st.spinner("🔄 Loading AI models..."):
                try:
                    csrnet = load_trained_model(model_path)
                    yolo = load_yolo_model()
                    st.session_state.csrnet = csrnet
                    st.session_state.yolo = yolo
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
                    return
        else:
            st.error(f"❌ Model file not found: {model_path}")
            st.info("📥 Download model from: https://drive.google.com/file/d/160AGUNDGEwVHEraYpwS7onyBNPWfySrh/view?usp=drive_link")
            return
    
    # WebRTC Live Detection
    st.markdown("## 📹 Live Camera Detection (WebRTC)")
    st.info("✅ **Browser-based camera access** - Works on Streamlit Cloud!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create video processor
        ctx = webrtc_streamer(
            key="crowd-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=CrowdVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Update processor settings
        if ctx.video_processor:
            ctx.video_processor.counter = AdaptiveHybridCounter(st.session_state.csrnet, st.session_state.yolo)
            ctx.video_processor.yolo_conf = yolo_conf
            ctx.video_processor.use_adaptive = use_adaptive
            ctx.video_processor.density_threshold = density_threshold
            ctx.video_processor.alert_threshold = alert_threshold
    
    with col2:
        st.markdown("### 📊 Live Stats")
        if ctx.video_processor:
            st.metric("Frames Processed", ctx.video_processor.frame_count)
            st.metric("Alert Threshold", alert_threshold)
            st.metric("YOLO Confidence", f"{yolo_conf:.2f}")
        
        st.markdown("### 🎮 Controls")
        if st.button("🔄 Reset Tracker"):
            if ctx.video_processor and ctx.video_processor.counter:
                ctx.video_processor.counter.reset_tracker()
                st.success("✅ Tracker reset!")
        
        st.markdown("### ℹ️ Info")
        st.write("""
        **How it works:**
        1. Click camera icon above
        2. Allow browser camera access
        3. Real-time detection starts!
        
        **Features:**
        - ✅ Cloud deployment
        - ✅ Browser camera
        - ✅ Real-time counting
        - ✅ Adaptive AI
        """)


if __name__ == "__main__":
    main()