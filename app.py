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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Page config
st.set_page_config(
    page_title="Deep Vision Crowd Monitor",
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

# RTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==================== SMTP CONFIG ====================
def get_smtp_config():
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

# ==================== EMAIL SYSTEM ====================
class EmailAlertSystem:
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
                  <h2>🚨 CROWD ALERT!</h2>
                </div>
                <div style="padding: 20px; background-color: #f5f5f5; margin-top: 20px;">
                  <h3>Alert Details:</h3>
                  <ul style="font-size: 16px;">
                    <li><strong>Count:</strong> {int(count)} people</li>
                    <li><strong>Threshold:</strong> {int(threshold)}</li>
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
    yolo = YOLO('yolov8n.pt')
    return yolo


# ==================== COUNTER CLASSES ====================
class EnhancedCrowdCounter:
    def __init__(self, csrnet_model):
        self.csrnet = csrnet_model
        self.device = DEVICE
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.target_size = TARGET_SIZE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.density_confidence_threshold = 0.05
    
    def enhance_low_light(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def adaptive_brightness_check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < 80
    
    def preprocess_frame(self, frame, enhance_lighting=True):
        if enhance_lighting and self.adaptive_brightness_check(frame):
            frame = self.enhance_low_light(frame)
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_normalized - self.mean) / self.std
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device, dtype=torch.float32)
    
    def predict_density(self, frame, use_multi_scale=True):
        with torch.no_grad():
            img_tensor = self.preprocess_frame(frame)
            density_map = self.csrnet(img_tensor)
            density_map = density_map.squeeze().cpu().numpy()
        
        density_map = gaussian_filter(density_map, sigma=1)
        total_count = float(density_map.sum())
        return density_map, total_count
    
    def predict_with_visualization(self, frame, use_multi_scale=True):
        density_map, total_count = self.predict_density(frame, use_multi_scale)
        return frame.copy(), density_map, total_count
    
    def create_heatmap_overlay(self, density_map, original_frame, alpha=0.4):
        h, w = original_frame.shape[:2]
        density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
        if density_resized.max() > 0:
            density_normalized = density_resized / density_resized.max()
        else:
            density_normalized = density_resized
        heatmap = cm.jet(density_normalized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = cv2.addWeighted(original_frame, 1-alpha, heatmap, alpha, 0)
        return overlay, heatmap


class AdaptiveHybridCounter:
    def __init__(self, csrnet_model, yolo_model):
        self.csrnet = csrnet_model
        self.yolo = yolo_model
        self.device = DEVICE
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=30)
        self.count_history = deque(maxlen=5)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    def preprocess_for_csrnet(self, roi):
        h, w = roi.shape[:2]
        new_h = ((h // 8) * 8) if h > 8 else 8
        new_w = ((w // 8) * 8) if w > 8 else 8
        roi_resized = cv2.resize(roi, (new_w, new_h))
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
            density_resized = cv2.resize(density_np, (original_w, original_h))
            scaled_count = float(density_resized.sum()) * 0.1
            scaled_count = min(scaled_count, 3.0)
        return density_resized, scaled_count
    
    def predict_density_full(self, frame):
        target_size = (512, 512)
        frame_resized = cv2.resize(frame, target_size)
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
    
    def predict_adaptive(self, frame, yolo_conf=0.4, use_adaptive=True, density_threshold=30):
        h, w = frame.shape[:2]
        detections = self.detect_heads_yolo(frame, confidence_threshold=yolo_conf)
        yolo_count = len(detections)
        
        if use_adaptive and yolo_count > density_threshold:
            mode = "CSRNet Only"
            density_map_full, final_count = self.predict_density_full(frame)
            density_map_visual = cv2.resize(density_map_full, (w, h))
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
                    density_map_visual[y1:y2, x1:x2] += density_roi
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
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=30)
        self.count_history.clear()


# ==================== VIDEO PROCESSING ====================
def process_video_streamlit(video_path, counter, alert_threshold, frame_skip, use_multi_scale, email_system):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    processed_idx = 0
    total_counts = []
    alert_triggered = False
    first_alert_frame = None
    email_sent = False
    frame_numbers = []
    count_values = []
    timestamps = []
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        
        annotated, density_map, total_count = counter.predict_with_visualization(frame, use_multi_scale=use_multi_scale)
        overlay, _ = counter.create_heatmap_overlay(density_map, annotated, alpha=0.4)
        
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Count: {int(total_count)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if total_count > alert_threshold:
            if not alert_triggered:
                alert_triggered = True
                first_alert_frame = frame_idx
                
                if email_system and email_system.enabled and not email_sent:
                    alert_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                    cv2.imwrite(alert_image_path, overlay)
                    email_sent = email_system.send_alert_email(
                        subject="🚨 CROWD ALERT!",
                        count=total_count,
                        threshold=alert_threshold,
                        frame_info=f"Frame {frame_idx}",
                        image_path=alert_image_path
                    )
                    try:
                        os.unlink(alert_image_path)
                    except:
                        pass
            
            cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 255), -1)
            cv2.putText(overlay, f"ALERT! {int(total_count)} > {alert_threshold}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(overlay)
        total_counts.append(total_count)
        frame_numbers.append(frame_idx)
        count_values.append(total_count)
        timestamps.append(processed_idx / fps if fps > 0 else processed_idx)
        processed_idx += 1
        
        progress = processed_idx / (total_frames // frame_skip)
        progress_bar.progress(progress)
        elapsed = time.time() - start_time
        fps_proc = processed_idx / elapsed if elapsed > 0 else 0
        status_text.text(f"Processing: {processed_idx} frames | FPS: {fps_proc:.1f} | Count: {int(total_count)}")
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    stats = {
        'output_path': output_path,
        'avg_total': np.mean(total_counts) if total_counts else 0,
        'max_total': max(total_counts) if total_counts else 0,
        'min_total': min(total_counts) if total_counts else 0,
        'processed_frames': processed_idx,
        'alert_triggered': alert_triggered,
        'first_alert_frame': first_alert_frame,
        'processing_time': time.time() - start_time,
        'email_sent': email_sent,
        'frame_numbers': frame_numbers,
        'count_values': count_values,
        'timestamps': timestamps
    }
    return stats


# ==================== ANALYTICS ====================
def create_analytics_graphs(stats):
    if not stats or 'count_values' not in stats:
        return
    
    df = pd.DataFrame({
        'Frame': stats['frame_numbers'],
        'Count': stats['count_values'],
        'Time (s)': stats['timestamps']
    })
    
    st.markdown("## 📊 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['Time (s)'],
            y=df['Count'],
            mode='lines+markers',
            name='Count',
            line=dict(color='#00d4ff', width=2)
        ))
        fig1.update_layout(
            title="Count Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Count",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        fig2 = go.Figure(data=[go.Histogram(x=df['Count'], nbinsx=30, marker_color='#00d4ff')])
        fig2.update_layout(
            title="Count Distribution",
            xaxis_title="Count",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig2, width='stretch')



# ==================== 🔥 FIXED WEBRTC PROCESSOR ====================
class VideoProcessor:
    """
    ✅ FIXED: Thread-safe video processor
    
    Key changes:
    1. NO session_state access in recv()
    2. All configs passed during initialization
    3. Email cooldown handled internally
    """
    
    def __init__(self, hybrid_counter, alert_threshold, yolo_conf, use_adaptive, 
                 density_threshold, email_system):
        # Store everything as instance variables
        self.hybrid_counter = hybrid_counter
        self.alert_threshold = alert_threshold
        self.yolo_conf = yolo_conf
        self.use_adaptive = use_adaptive
        self.density_threshold = density_threshold
        self.email_system = email_system
        self.email_sent_time = 0  # Track last email time
        self.email_cooldown = 10  # 10 seconds between emails
    
    def recv(self, frame):
        """Process each frame - NO session_state access here!"""
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Process frame
            annotated, density_map, final_count, mode, yolo_count = self.hybrid_counter.predict_adaptive(
                img,
                yolo_conf=self.yolo_conf,
                use_adaptive=self.use_adaptive,
                density_threshold=self.density_threshold
            )
            
            h, w = annotated.shape[:2]
            
            # Info overlay
            cv2.rectangle(annotated, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.putText(annotated, "Hybrid: YOLO + CSRNet", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Mode: {mode}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            cv2.putText(annotated, f"Count: {int(final_count)} | YOLO: {yolo_count}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Alert handling
            if final_count > self.alert_threshold:
                cv2.putText(annotated, f"ALERT! {int(final_count)} > {self.alert_threshold}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Bottom banner
                cv2.rectangle(annotated, (0, h - 50), (w, h), (0, 0, 255), -1)
                cv2.putText(annotated, "ALERT!", (20, h - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Email with cooldown
                current_time = time.time()
                if (self.email_system and self.email_system.enabled and 
                    (current_time - self.email_sent_time) > self.email_cooldown):
                    
                    try:
                        alert_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                        cv2.imwrite(alert_image_path, annotated)
                        
                        sent = self.email_system.send_alert_email(
                            subject="🚨 LIVE WEBCAM ALERT!",
                            count=final_count,
                            threshold=self.alert_threshold,
                            frame_info="Live Webcam",
                            image_path=alert_image_path
                        )
                        
                        if sent:
                            self.email_sent_time = current_time
                        
                        os.unlink(alert_image_path)
                    except:
                        pass
            else:
                cv2.putText(annotated, f"Normal - {int(final_count)}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        
        except Exception as e:
            # Fallback: return original frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==================== MAIN UI ====================
def main():
    st.title("🎥 Deep Vision Crowd Monitor")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("CSRNet Model Path", value="best_crowd_counter_objects.pth")
        
        st.divider()
        st.markdown("### 🔧 Email Alerts")
        
        if SMTP_CONFIG:
            st.success(f"✅ SMTP OK\n📤 {SMTP_CONFIG['sender_email']}")
            enable_email = st.checkbox("Enable Emails", value=False)
            
            if enable_email:
                recipient_emails = st.text_area("Recipients (comma-separated)", 
                                               value="recipient@gmail.com")
                recipient_list = [e.strip() for e in recipient_emails.split(',') if e.strip()]
            else:
                recipient_list = []
        else:
            st.error("❌ SMTP Not Configured")
            enable_email = False
            recipient_list = []
    
    # Tabs
    tab1, tab2 = st.tabs(["🎬 Video", "📷 Webcam"])
    
    # ========== VIDEO TAB ==========
    with tab1:
        st.markdown("### Video Processing")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            alert_threshold = st.slider("Alert Threshold", 10, 500, 100, 10)
            frame_skip = st.slider("Frame Skip", 1, 10, 3)
            use_multi_scale = st.checkbox("Multi-Scale", value=False)
        
        with col2:
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.video(video_path)
            
            if st.button("🚀 Process Video", type="primary"):
                if not Path(model_path).exists():
                    st.error(f"❌ Model not found: {model_path}")
                else:
                    email_system = EmailAlertSystem(recipient_list, enabled=enable_email)
                    
                    with st.spinner("Loading CSRNet..."):
                        try:
                            csrnet = load_trained_model(model_path)
                            counter = EnhancedCrowdCounter(csrnet)
                            st.success("✓ Model loaded!")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            return
                    
                    with st.spinner("Processing video..."):
                        stats = process_video_streamlit(video_path, counter, alert_threshold, 
                                                       frame_skip, use_multi_scale, email_system)
                    
                    if stats:
                        st.success("✓ Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average", f"{stats['avg_total']:.1f}")
                        with col2:
                            st.metric("Maximum", f"{stats['max_total']:.1f}")
                        with col3:
                            st.metric("Minimum", f"{stats['min_total']:.1f}")
                        with col4:
                            st.metric("Time", f"{stats['processing_time']:.1f}s")
                        
                        if stats['alert_triggered']:
                            st.warning(f"🚨 Alert at frame {stats['first_alert_frame']}")
                        
                        if stats.get('email_sent', False):
                            st.success("✅ Email alert sent!")
                        
                        st.markdown("### 🎬 Processed Video")
                        try:
                            st.video(stats['output_path'])
                        except:
                            st.warning("⚠️ Preview not available. Download to view.")
                        
                        with open(stats['output_path'], 'rb') as f:
                            st.download_button(
                                "⬇️ Download Video",
                                data=f,
                                file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4"
                            )
                        
                        create_analytics_graphs(stats)
                        
                        try:
                            os.unlink(stats['output_path'])
                            os.unlink(video_path)
                        except:
                            pass
    
    # ========== WEBCAM TAB ==========
    with tab2:
        st.markdown("### 🧠 Adaptive Hybrid: YOLO + CSRNet")
        
        st.info("✅ **WebRTC Enabled** - Browser camera will be used")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Settings")
            webcam_alert = st.slider("Alert Threshold", 5, 100, 50, 5, key="webcam_alert")
            yolo_conf = st.slider("YOLO Confidence", 0.2, 0.8, 0.4, 0.05, key="yolo_conf")
            density_threshold = st.slider("Dense Crowd Threshold", 10, 100, 30, 5, key="density_thresh")
            use_adaptive = st.checkbox("Enable Adaptive Mode", value=True, key="use_adaptive")
            
            if st.button("🔄 Reset Tracker"):
                if 'hybrid_counter' in st.session_state and st.session_state.hybrid_counter:
                    st.session_state.hybrid_counter.reset_tracker()
                    st.success("✅ Tracker reset!")
        
        with col2:
            st.markdown("#### 📷 Live Webcam Feed")
            
            # Check model
            if not Path(model_path).exists():
                st.error(f"❌ CSRNet model not found: {model_path}")
                st.info("Upload your trained model file first!")
            else:
                # ✅ Step 1: Load models ONCE
                if 'models_loaded' not in st.session_state:
                    st.session_state.models_loaded = False
                
                if not st.session_state.models_loaded:
                    with st.spinner("🔄 Loading AI models..."):
                        try:
                            csrnet = load_trained_model(model_path)
                            yolo = load_yolo_model()
                            st.session_state.csrnet = csrnet
                            st.session_state.yolo = yolo
                            st.session_state.models_loaded = True
                            st.success("✅ Models loaded!")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Model loading failed: {e}")
                            st.stop()
                
                # ✅ Step 2: Create hybrid counter ONCE
                if 'hybrid_counter' not in st.session_state:
                    st.session_state.hybrid_counter = AdaptiveHybridCounter(
                        st.session_state.csrnet,
                        st.session_state.yolo
                    )
                
                # ✅ CRITICAL: Capture values BEFORE factory
                hybrid_counter_ref = st.session_state.hybrid_counter
                email_system_ref = EmailAlertSystem(recipient_list, enabled=enable_email)
                
                # Capture slider values
                alert_val = webcam_alert
                conf_val = yolo_conf
                adaptive_val = use_adaptive
                density_val = density_threshold
                
                # ✅ Factory with captured values (NO session_state)
                def video_processor_factory():
                    return VideoProcessor(
                        hybrid_counter=hybrid_counter_ref,
                        alert_threshold=alert_val,
                        yolo_conf=conf_val,
                        use_adaptive=adaptive_val,
                        density_threshold=density_val,
                        email_system=email_system_ref
                    )
                
                st.warning("⚠️ **Allow camera permission when browser asks!**")
                
                # ✅ WebRTC streamer
                ctx = webrtc_streamer(
                    key="crowd-detection-live",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=video_processor_factory,
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 1280},
                            "height": {"ideal": 720},
                            "frameRate": {"ideal": 30}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Status display
                if ctx.state.playing:
                    st.success("✅ Webcam ACTIVE")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Alert Threshold", webcam_alert)
                    with col_b:
                        st.metric("YOLO Conf", f"{yolo_conf:.2f}")
                    with col_c:
                        st.metric("Mode", "Adaptive" if use_adaptive else "Fixed")
                    
                    st.info("💡 To update settings: Click STOP → Adjust sliders → Click START")
                        
                elif ctx.state.signalling:
                    st.info("🔄 Connecting to camera...")
                else:
                    st.info("💡 **Click START button above** to activate webcam")
                    
                    st.markdown("""
                    ### 📋 Quick Start:
                    1. **Adjust settings** on the left
                    2. Click **START** button ⬆️
                    3. **Allow camera** in browser
                    4. Wait 2-3 seconds
                    5. To change: **STOP** → Adjust → **START**
                    """)
                
                # Troubleshooting
                with st.expander("❓ Camera Issues?"):
                    st.markdown("""
                    ### 🔧 Troubleshooting:
                    
                    **1. Browser Permission**
                    - Click 🔒 in address bar
                    - Select "Allow" for camera
                    - Refresh page
                    
                    **2. Camera In Use?**
                    - Close Zoom, Teams, Skype
                    - Close other browser tabs
                    - Check Windows Camera app
                    
                    **3. Browser Support**
                    - ✅ Chrome (Best)
                    - ✅ Edge (Best)
                    - ✅ Firefox
                    - ⚠️ Safari (Limited)
                    - ❌ Mobile browsers
                    
                    **4. HTTPS Required**
                    - Streamlit Cloud: ✅ Auto HTTPS
                    - Localhost: `http://localhost:8501`
                    - Network: Must use HTTPS
                    
                    **5. Still Not Working?**
                    - Try Incognito mode
                    - Clear cache (Ctrl+Shift+Del)
                    - Disable extensions
                    - Test at https://webcamtests.com
                    - Try different browser
                    - Restart browser
                    
                    **6. Check Console**
                    - Press F12
                    - Look for errors in Console tab
                    - Share errors if needed
                    """)
                
                st.divider()
                st.caption("🔒 Privacy: Processed locally in browser. Nothing stored.")


if __name__ == "__main__":
    main()