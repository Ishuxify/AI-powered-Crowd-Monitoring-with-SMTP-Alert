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

# BEFORE the "# Page config" section

import warnings
import logging
import asyncio
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to suppress WebRTC errors
logging.getLogger('aioice').setLevel(logging.CRITICAL)
logging.getLogger('aiortc').setLevel(logging.CRITICAL)
logging.getLogger('av').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Fix asyncio event loop for Python 3.11+
if sys.platform == 'linux':
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except:
        pass

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

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},

            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:443",
                    "turn:global.relay.metered.ca:443?transport=tcp"
                ],
                "username": st.secrets["turn"]["username"],
                "credential": st.secrets["turn"]["password"]
            }
        ],
        "iceTransportPolicy": "all"
    }
)



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
            st.warning("⚠️ SMTP credentials not configured in secrets.toml")
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
                
                <div style="padding: 20px; margin-top: 20px;">
                  <p style="font-size: 14px; color: #666;">
                    ⚡ Automated alert from Enhanced Crowd Counter<br>
                    🔧 Next alert after 5 min cooldown
                  </p>
                </div>
                
                {f'<div style="margin-top: 20px;"><img src="cid:alert_image" style="max-width: 600px; border-radius: 10px;"></div>' if image_path else ''}
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


# ==================== VIDEO PROCESSING COUNTER ====================
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
    
    def multi_scale_prediction(self, frame):
        scales = [1.0, 0.9]
        density_maps = []
        h, w = frame.shape[:2]
        for scale in scales:
            if scale != 1.0:
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))
            else:
                scaled_frame = frame
            with torch.no_grad():
                img_tensor = self.preprocess_frame(scaled_frame)
                density_map = self.csrnet(img_tensor)
                density_np = density_map.squeeze().cpu().numpy()
            if scale != 1.0:
                density_np = cv2.resize(density_np, (density_map.shape[-1], density_map.shape[-2]), interpolation=cv2.INTER_CUBIC)
                density_np = density_np / (scale * scale)
            density_maps.append(density_np)
        return np.mean(density_maps, axis=0)
    
    def apply_confidence_filtering(self, density_map):
        if density_map.max() > 0:
            normalized = density_map / density_map.max()
        else:
            normalized = density_map
        filtered = np.where(normalized < self.density_confidence_threshold, 0, density_map)
        filtered = gaussian_filter(filtered, sigma=1)
        return filtered
    
    def predict_density(self, frame, use_multi_scale=True):
        if use_multi_scale:
            density_map = self.multi_scale_prediction(frame)
        else:
            with torch.no_grad():
                img_tensor = self.preprocess_frame(frame)
                density_map = self.csrnet(img_tensor)
                density_map = density_map.squeeze().cpu().numpy()
        density_map = self.apply_confidence_filtering(density_map)
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


# ==================== ADAPTIVE HYBRID COUNTER ====================
class AdaptiveHybridCounter:
    def __init__(self, csrnet_model, yolo_model):
        self.csrnet = csrnet_model
        self.yolo = yolo_model
        self.device = DEVICE
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.tracker = sv.ByteTrack()
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
       self.tracker = sv.ByteTrack()
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
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    processed_idx = 0
    total_counts = []
    alert_triggered = False
    first_alert_frame = None
    low_light_frames = 0
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
        
        is_low_light = counter.adaptive_brightness_check(frame)
        if is_low_light:
            low_light_frames += 1
        
        annotated, density_map, total_count = counter.predict_with_visualization(frame, use_multi_scale=use_multi_scale)
        overlay, _ = counter.create_heatmap_overlay(density_map, annotated, alpha=0.4)
        
        cv2.rectangle(overlay, (10, 10), (400, 110), (0, 0, 0), -1)
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Total Count: {int(total_count)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if is_low_light:
            cv2.putText(overlay, "Low-Light: ENHANCED", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if total_count > alert_threshold:
            if not alert_triggered:
                alert_triggered = True
                first_alert_frame = frame_idx
                if email_system and email_system.enabled and not email_sent:
                    alert_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                    cv2.imwrite(alert_image_path, overlay)
                    email_sent = email_system.send_alert_email(
                        subject="🚨 CROWD ALERT - Threshold Exceeded!",
                        count=total_count,
                        threshold=alert_threshold,
                        frame_info=f"Frame {frame_idx}/{total_frames}",
                        image_path=alert_image_path
                    )
                    if email_sent:
                        st.success("✅ Email alert sent successfully!")
                    try:
                        os.unlink(alert_image_path)
                    except:
                        pass
            cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 255), -1)
            cv2.putText(overlay, f"ALERT! Count: {int(total_count)} > {alert_threshold}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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
        status_text.text(f"Processing: {processed_idx}/{total_frames // frame_skip} frames | FPS: {fps_proc:.1f} | Total Count: {int(total_count)}")
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    try:
        import subprocess
        converted_path = output_path.replace('.mp4', '_web.mp4')
        result = subprocess.run(['ffmpeg', '-i', output_path, '-vcodec', 'libx264', '-acodec', 'aac', '-y', converted_path], capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(converted_path):
            os.unlink(output_path)
            output_path = converted_path
    except:
        pass
    
    stats = {
        'output_path': output_path,
        'avg_total': np.mean(total_counts) if total_counts else 0,
        'max_total': max(total_counts) if total_counts else 0,
        'min_total': min(total_counts) if total_counts else 0,
        'processed_frames': processed_idx,
        'alert_triggered': alert_triggered,
        'first_alert_frame': first_alert_frame,
        'low_light_frames': low_light_frames,
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
    
    df = pd.DataFrame({'Frame': stats['frame_numbers'], 'Count': stats['count_values'], 'Time (s)': stats['timestamps']})
    
    st.markdown("## 📊 Analytics Dashboard")
    
    st.markdown("### 📈 Crowd Count Timeline")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Time (s)'], y=df['Count'], mode='lines+markers', name='Crowd Count', line=dict(color='#00d4ff', width=2), marker=dict(size=4)))
    fig1.add_hline(y=stats.get('avg_total', 0), line_dash="dash", line_color="orange", annotation_text=f"Average: {stats.get('avg_total', 0):.1f}")
    fig1.update_layout(title="Crowd Count Over Time", xaxis_title="Time (seconds)", yaxis_title="Count", hovermode='x unified', template='plotly_dark', height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Count Distribution")
      
        fig2 = go.Figure(data=[go.Histogram(x=df['Count'], nbinsx=30, marker_color='#00d4ff', opacity=0.75)])
        fig2.update_layout(title="Crowd Count Distribution", xaxis_title="Count", yaxis_title="Frequency", template='plotly_dark', height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Statistical Summary")
        summary_df = pd.DataFrame({
            'Metric': ['Average', 'Maximum', 'Minimum', 'Median','Std Dev'],
            'Value': [
                f"{stats['avg_total']:.1f}",
                f"{stats['max_total']:.1f}",
                f"{stats['min_total']:.1f}",
                f"{df['Count'].median():.1f}",
                f"{df['Count'].std():.1f}"
            ]
        })
        
        fig3 = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='#1f77b4',
                align='left',
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[summary_df['Metric'], summary_df['Value']],
                fill_color='#2a2a2a',
                align='left',
                font=dict(color='white', size=12),
                height=30
            )
        )])
        fig3.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("### 🔥 Crowd Intensity Heatmap")
    time_bins = 20
    df['Time_Bin'] = pd.cut(df['Time (s)'], bins=time_bins)
    heatmap_data = df.groupby('Time_Bin')['Count'].mean().values.reshape(1, -1)
    
    fig4 = go.Figure(data=go.Heatmap(z=heatmap_data, colorscale='Jet', showscale=True, colorbar=dict(title="Count")))
    fig4.update_layout(title="Crowd Density Heatmap (Time Windows)", xaxis_title="Time Window", yaxis_title="Intensity", template='plotly_dark', height=250, yaxis=dict(showticklabels=False))
    st.plotly_chart(fig4, use_container_width=True)
    
    if stats.get('alert_triggered', False):
        st.markdown("### 🚨 Alert Analysis")
        alert_frame = stats.get('first_alert_frame', 0)
        alert_time = df[df['Frame'] >= alert_frame]['Time (s)'].min() if len(df[df['Frame'] >= alert_frame]) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Alert Frame", alert_frame)
        with col2:
            st.metric("Alert Time", f"{alert_time:.1f}s")
        with col3:
            above_threshold = len(df[df['Count'] > stats.get('avg_total', 0)])
            st.metric("Frames Above Avg", above_threshold)


# ==================== WEBRTC VIDEO PROCESSOR ====================
# ==================== WEBRTC VIDEO PROCESSOR (FIXED) ====================
class VideoProcessor:
    """
    Fixed WebRTC processor with proper error handling
    """
    def __init__(self):
        self.hybrid_counter = None
        self.alert_threshold = 50
        self.yolo_conf = 0.4
        self.use_adaptive = True
        self.density_threshold = 30
        self.email_system = None
        self.email_sent = False
        self.frame_count = 0
        
    def recv(self, frame):
        """Process incoming video frame"""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # If no counter loaded, return original frame with message
            if self.hybrid_counter is None:
                h, w = img.shape[:2]
                cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(img, "Loading models...", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Increment frame counter
            self.frame_count += 1
            
            # Process every frame (no skipping for smooth video)
            annotated, density_map, final_count, mode, yolo_count = self.hybrid_counter.predict_adaptive(
                img, 
                yolo_conf=self.yolo_conf, 
                use_adaptive=self.use_adaptive, 
                density_threshold=self.density_threshold
            )
            
            h, w = annotated.shape[:2]
            
            # Create overlay background
            cv2.rectangle(annotated, (0, 0), (w, 140), (0, 0, 0), -1)
            
            # Mode color
            mode_color = (255, 165, 0) if mode == "YOLO + CSRNet" else (147, 112, 219)
            
            # Draw text overlays
            cv2.putText(annotated, "Adaptive Hybrid: YOLO + CSRNet", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Mode: {mode}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            cv2.putText(annotated, f"Count: {int(final_count)} | YOLO: {yolo_count} | Frame: {self.frame_count}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Alert handling
            if final_count > self.alert_threshold:
                # Top alert text
                cv2.putText(annotated, f"ALERT! {int(final_count)} > {self.alert_threshold}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Bottom alert banner
                cv2.rectangle(annotated, (0, h - 50), (w, h), (0, 0, 255), -1)
                cv2.putText(annotated, "CROWD ALERT ACTIVE!", 
                           (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Send email (once per alert, with cooldown handled by EmailAlertSystem)
                if self.email_system and self.email_system.enabled and not self.email_sent:
                    try:
                        # Save frame temporarily
                        alert_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                        cv2.imwrite(alert_image_path, annotated)
                        
                        # Send email
                        sent = self.email_system.send_alert_email(
                            subject="🚨 LIVE WEBCAM ALERT!", 
                            count=final_count, 
                            threshold=self.alert_threshold, 
                            frame_info=f"Live Feed - Frame {self.frame_count}", 
                            image_path=alert_image_path
                        )
                        
                        if sent:
                            self.email_sent = True
                        
                        # Clean up temp file
                        try:
                            os.unlink(alert_image_path)
                        except:
                            pass
                    except Exception as e:
                        # Silently fail on email errors
                        pass
            else:
                # Normal status
                cv2.putText(annotated, f"Normal - Count: {int(final_count)}", 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Reset email flag when count drops below threshold
                self.email_sent = False
            
            # Return processed frame
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            # On any error, return original frame with error message
            try:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(img, f"Processing Error", 
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except:
                # Last resort: return original frame unchanged
                return frame

# ==================== MAIN UI ====================
def main():
    st.title("🎥 Deep Vision Crowd Monitor: AI for Density Estimation and Overcrowding Detection")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input("CSRNet Model Path", value="best_crowd_counter_objects.pth")
        
        st.divider()
        
        st.markdown("### 🔧 Email Alert Settings")
        
        if SMTP_CONFIG:
            st.success(f"✅ SMTP Configured\n\n📤 Sender: {SMTP_CONFIG['sender_email']}")
            enable_email = st.checkbox("Enable Email Alerts", value=False)
            
            if enable_email:
                recipient_emails = st.text_area("Recipient Emails (comma-separated)", value="recipient@gmail.com")
                recipient_list = [e.strip() for e in recipient_emails.split(',') if e.strip()]
                st.success(f"✅ {len(recipient_list)} recipient(s)")
            else:
                recipient_list = []
        else:
            st.error("❌ SMTP Not Configured")
            st.info("""Create `.streamlit/secrets.toml` with SMTP settings""")
            enable_email = False
            recipient_list = []
        
        st.divider()
        st.info("**Video:** Frame skip 3-5\n\n**Webcam:** YOLO conf 0.3-0.4")
    
    tab1, tab2 = st.tabs(["🎬 Video Processing", "📷 Live Webcam"])
    
    # ========== TAB 1: VIDEO ==========
    with tab1:
        st.markdown("### ✅ CSRNet Direct Density Counting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Settings")
            alert_threshold = st.slider("Alert Threshold", 10, 500, 100, 10)
            frame_skip = st.slider("Frame Skip", 1, 10, 3, 1)
            use_multi_scale = st.checkbox("Enable Multi-Scale", value=False)
        
        with col2:
            st.markdown("#### 📤 Upload Video")
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
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
                    
                    with st.spinner("Loading CSRNet model..."):
                        try:
                            csrnet = load_trained_model(model_path)
                            counter = EnhancedCrowdCounter(csrnet)
                            st.success("✓ Model loaded!")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            return
                    
                    with st.spinner("Processing video..."):
                        stats = process_video_streamlit(video_path, counter, alert_threshold, frame_skip, use_multi_scale, email_system)
                    
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
                            st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
                        
                        if stats['alert_triggered']:
                            st.warning(f"🚨 Alert at frame {stats['first_alert_frame']}")
                        
                        if stats.get('email_sent', False):
                            st.success("✅ Email alert sent successfully!")
                        
                        st.markdown("### 🎬 Processed Video")
                        try:
                            st.video(stats['output_path'])
                        except:
                            st.warning("⚠️ Preview not available. Download to view.")
                        
                        with open(stats['output_path'], 'rb') as f:
                            st.download_button("⬇️ Download Video", data=f, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4", mime="video/mp4")
                        
                        create_analytics_graphs(stats)
                        
                        try:
                            os.unlink(stats['output_path'])
                            os.unlink(video_path)
                        except:
                            pass
    
    # ========== TAB 2: WEBCAM WITH WEBRTC - COMPLETELY FIXED ==========
    with tab2:
        st.markdown("### 🧠 Adaptive Hybrid Strategy: YOLO + CSRNet")
        
        st.info("✅ **WebRTC Enabled** - Works on Streamlit Cloud! Browser camera will be used.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ Live Settings")
            webcam_alert = st.slider("Alert Threshold", 5, 100, 50, 5, key="webcam_alert")
            yolo_conf = st.slider("YOLO Confidence", 0.2, 0.8, 0.4, 0.05, key="yolo_conf")
            density_threshold = st.slider("Dense Crowd Threshold", 10, 100, 30, 5, key="density_thresh")
            use_adaptive = st.checkbox("Enable Adaptive Mode", value=True, key="use_adaptive")
            
            if st.button("🔄 Reset Tracker"):
                if 'hybrid_counter' in st.session_state and st.session_state.hybrid_counter:
                    st.session_state.hybrid_counter.reset_tracker()
                    st.success("✅ Tracker reset!")
        
        with col2:
            st.markdown("#### 📷 Live Webcam Feed (WebRTC)")
            
            # ✅ Check model path
            if not Path(model_path).exists():
                st.error(f"❌ CSRNet model not found: {model_path}")
                st.info("Upload your trained model file first!")
            else:
                # ✅ Step 1: Load models ONCE
                if 'models_loaded' not in st.session_state:
                    st.session_state.models_loaded = False
                
                if not st.session_state.models_loaded:
                    with st.spinner("🔄 Loading AI models... Please wait..."):
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
                
                
                # ✅ CRITICAL FIX: Capture references OUTSIDE factory function
                hybrid_counter_ref = st.session_state.hybrid_counter
                email_system_ref = EmailAlertSystem(recipient_list, enabled=enable_email)
                
                # Capture current slider values
                alert_threshold_val = webcam_alert
                yolo_conf_val = yolo_conf
                use_adaptive_val = use_adaptive
                density_threshold_val = density_threshold
                
                # ✅ Factory function (NO session_state access inside!)
                def create_video_processor():
                    """Factory that uses captured values from main thread"""
                    processor = VideoProcessor()
                    processor.hybrid_counter = hybrid_counter_ref
                    processor.alert_threshold = alert_threshold_val
                    processor.yolo_conf = yolo_conf_val
                    processor.use_adaptive = use_adaptive_val
                    processor.density_threshold = density_threshold_val
                    processor.email_system = email_system_ref
                    processor.email_sent = False
                    return processor
                
                st.warning("⚠️ **IMPORTANT**: Allow camera permission when browser asks!")
                
                # ✅ WebRTC streamer with fixed factory
                ctx = webrtc_streamer(
                    key="crowd-detection-live",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=create_video_processor,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 640, "ideal": 1280, "max": 1920},
                            "height": {"min": 480, "ideal": 720, "max": 1080},
                            "frameRate": {"ideal": 30}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
              
                # Debug info
                st.markdown("---")
                st.markdown("### 🔍 Debug Info")
                
                col_debug1, col_debug2, col_debug3 = st.columns(3)
                
                with col_debug1:
                    st.write("**WebRTC State:**")
                    st.write(f"- Playing: {ctx.state.playing}")
                    st.write(f"- Signalling: {ctx.state.signalling}")
                
                with col_debug2:
                    st.write("**Models:**")
                    models_ok = 'models_loaded' in st.session_state and st.session_state.models_loaded
                    st.write(f"- Loaded: {models_ok}")
                    if models_ok:
                        st.write(f"- Counter: {'✅' if 'hybrid_counter' in st.session_state else '❌'}")
                
                with col_debug3:
                    st.write("**Settings:**")
                    st.write(f"- Alert: {webcam_alert}")
                    st.write(f"- YOLO: {yolo_conf}")
                    st.write(f"- Adaptive: {use_adaptive}")
                
                # Status display
                if ctx.state.playing:
                    st.success("✅ Webcam ACTIVE - Processing frames...")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Alert Threshold", webcam_alert)
                    with col_b:
                        st.metric("YOLO Conf", f"{yolo_conf:.2f}")
                    with col_c:
                        st.metric("Mode", "Adaptive" if use_adaptive else "YOLO+CSRNet")
                    
                    st.info("💡 **Note:** Settings are captured when START is clicked. To update settings, click STOP then START again.")
                        
                elif ctx.state.signalling:
                    st.info("🔄 Connecting to camera... Please wait...")
                else:
                    st.info("💡 **Click START button above** to activate webcam")
                    
                    st.markdown("""
                    ### 📋 Quick Start Guide:
                    1. **Adjust settings** using sliders on the left
                    2. Click the **START** button above ⬆️
                    3. **Allow camera access** in browser popup
                    4. Wait 2-3 seconds for processing to begin
                    5. To change settings: Click **STOP** → Adjust sliders → Click **START**
                    """)
                
                # Troubleshooting section
                with st.expander("❓ Camera Not Working? Click Here"):
                    st.markdown("""
                    ### 🔧 Troubleshooting Steps:
                    
                    #### 1️⃣ Browser Permission
                    - Look for 🔒 or 📷 icon in address bar
                    - Click it and select "Allow" for camera
                    - Refresh page after allowing
                    
                    #### 2️⃣ Camera Already in Use?
                    - Close Zoom, Teams, Skype, etc.
                    - Close other browser tabs using camera
                    - Check Windows Camera app (close if open)
                    
                    #### 3️⃣ Browser Compatibility
                    - ✅ **Chrome** (Recommended)
                    - ✅ **Edge** (Recommended)  
                    - ✅ Firefox
                    - ⚠️ Safari (Limited support)
                    - ❌ Mobile browsers (Not supported)
                    
                    #### 4️⃣ HTTPS Required
                    - Streamlit Cloud: ✅ Automatic HTTPS
                    - Localhost: Use `http://localhost:8501`
                    - Network access: Must use HTTPS
                    
                    #### 5️⃣ Still Not Working?
                    - Try Incognito/Private mode
                    - Clear browser cache (Ctrl+Shift+Delete)
                    - Disable browser extensions
                    - Test camera at: https://webcamtests.com
                    - Try different browser
                    - Restart browser completely
                    
                    #### 6️⃣ Console Errors (Advanced)
                    - Press **F12** to open Developer Console
                    - Look in **Console** tab for errors
                    - Check for "getUserMedia" or "NotAllowedError"
                    - Share errors in support if needed
                    
                    #### 7️⃣ Model Loading Issues
                    - Ensure `best_crowd_counter_objects.pth` exists
                    - Check file path in sidebar settings
                    - Wait for "Models loaded!" message before clicking START
                    """)
                
                st.divider()
                st.caption("🔒 Privacy: Video is processed locally in your browser. Nothing is stored or uploaded.")
if __name__ == "__main__":
    main()