# 🎥 Deep Vision Crowd Monitor

AI-powered real-time crowd density estimation and overcrowding detection system using CSRNet and YOLOv8.

## 🌟 Features

- **🎯 Dual Counting Modes**
  - CSRNet: Direct density map estimation for dense crowds
  - Adaptive Hybrid: YOLO + CSRNet for optimal accuracy
  
- **📹 Video Processing**
  - Upload and process pre-recorded videos
  - Multi-scale prediction for improved accuracy
  - Low-light enhancement with adaptive CLAHE
  - Automated alert detection

- **📷 Live Webcam Monitoring**
  - Real-time crowd counting
  - Adaptive switching between YOLO and CSRNet
  - ByteTrack integration for person tracking

- **📧 Email Alert System**
  - Automated threshold-based alerts
  - HTML email with snapshot attachments
  - 5-minute cooldown to prevent spam

- **📊 Analytics Dashboard**
  - Interactive Plotly visualizations
  - Count timeline and distribution
  - Statistical summaries
  - Crowd intensity heatmaps

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install streamlit opencv-python numpy torch torchvision ultralytics supervision matplotlib scipy pillow pandas plotly
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deep-vision-crowd-monitor.git
cd deep-vision-crowd-monitor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the trained CSRNet model**

Download `best_crowd_counter_objects.pth` from Google Drive:

🔗 **[Download CSRNet Model (78 MB)](https://drive.google.com/file/d/160AGUNDGEwVHEraYpwS7onyBNPWfySrh/view?usp=drive_link)**

Place it in the project root directory.

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
```
http://localhost:8501
```

## 📁 Project Structure

```
deep-vision-crowd-monitor/
│
├── app.py                              # Main Streamlit application
├── best_crowd_counter_objects.pth      # CSRNet model (download required)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
└── .streamlit/
    └── secrets.toml                    # SMTP credentials (optional)
```

## 🎮 Usage

### Video Processing Mode

1. Navigate to **"Video Processing"** tab
2. Configure settings:
   - Alert Threshold: 10-500 people
   - Frame Skip: 1-10 (higher = faster processing)
   - Multi-Scale: Enable for better accuracy
3. Upload video (MP4, AVI, MOV, MKV)
4. Click **"Process Video"**
5. Download processed video with overlays

### Live Webcam Mode

1. Navigate to **"Live Webcam"** tab
2. Configure settings:
   - Alert Threshold: 5-100 people
   - YOLO Confidence: 0.2-0.8
   - Dense Crowd Threshold: 10-100
   - Adaptive Mode: Auto-switch between models
3. Click **"Start Webcam"**
4. Real-time crowd monitoring begins

**Note:** Webcam mode requires local execution (does not work on Streamlit Cloud)

## 📧 Email Alerts Setup (Optional)

Create `.streamlit/secrets.toml`:

```toml
[smtp]
server = "smtp.gmail.com"
port = 587
sender_email = "your-email@gmail.com"
sender_password = "your-app-password"
```

### Gmail App Password Setup
1. Enable 2-Factor Authentication
2. Go to: https://myaccount.google.com/apppasswords
3. Generate app password
4. Use in `secrets.toml`

## 🧠 Models

### CSRNet (Crowd Density Estimation)
- **Architecture:** VGG16 frontend + dilated convolution backend
- **Input:** 512×512 RGB images
- **Output:** Density map (1/8 resolution)
- **Use Case:** Dense crowd scenes (30+ people)

### YOLOv8n (Object Detection)
- **Auto-downloads** on first run (~6 MB)
- **Classes:** 80 COCO classes (person detection)
- **Use Case:** Sparse crowds (< 30 people)

### Adaptive Hybrid Strategy
- Automatically switches between YOLO and CSRNet
- YOLO for sparse scenes (< 30 people)
- CSRNet for dense crowds (≥ 30 people)
- Combines detection boxes with density maps

## 📊 Performance

| Mode | Best For | FPS (CPU) | FPS (GPU) | Accuracy |
|------|----------|-----------|-----------|----------|
| CSRNet Only | Dense crowds | 0.5-2 | 10-30 | High density |
| YOLO + CSRNet | Sparse crowds | 5-15 | 30-60 | High precision |
| Adaptive Hybrid | All scenarios | 3-10 | 15-40 | Balanced |

## 🛠️ Troubleshooting

**Model not found error:**
- Verify `best_crowd_counter_objects.pth` is in project root
- Check file permissions

**Webcam not working:**
- Webcam mode only works on local machines
- Does not work on Streamlit Cloud deployments

**Low accuracy:**
- Enable multi-scale prediction
- Adjust YOLO confidence threshold (0.3-0.5)
- Check lighting conditions (auto-enhancement enabled)

## 🔧 Configuration

**In Sidebar:**
- CSRNet Model Path (default: `best_crowd_counter_objects.pth`)
- Email Alert Settings
- SMTP Configuration

**Video Processing:**
- Alert Threshold
- Frame Skip (performance vs accuracy)
- Multi-Scale Prediction

**Webcam Mode:**
- Alert Threshold
- YOLO Confidence
- Dense Crowd Threshold
- Adaptive Mode Toggle

## 📚 Technologies Used

- **Streamlit** - Web interface
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing
- **YOLOv8 (Ultralytics)** - Object detection
- **Supervision** - Tracking utilities
- **Plotly** - Interactive visualizations
- **SMTP** - Email alerts

