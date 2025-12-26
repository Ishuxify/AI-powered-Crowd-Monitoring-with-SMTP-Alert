# Model Setup Guide - Deep Vision Crowd Monitor

## Required Files

### 1. CSRNet Model
**File needed:** `best_crowd_counter_objects.pth`
- Model download link : https://drive.google.com/file/d/160AGUNDGEwVHEraYpwS7onyBNPWfySrh/view?usp=drive_link

- Place this file in the same folder as `app.py`
- This is your trained crowd counting model
- File size: typically 50-200 MB

### 2. YOLOv8 Model
**File:** `yolov8n.pt`

- **Auto-downloads** when you first run the webcam feature
- No manual setup needed

---

## Installation

```bash
pip install streamlit opencv-python numpy torch torchvision ultralytics supervision matplotlib scipy pillow pandas plotly
```

---

## Running the App

1. Make sure `best_crowd_counter_objects.pth` is in your project folder
2. Run:
```bash
streamlit run app.py
```
3. Open browser: `http://localhost:8501`

---

## Optional: Email Alerts

Create `.streamlit/secrets.toml`:

```toml
[smtp]
server = "smtp.gmail.com"
port = 587
sender_email = "your-email@gmail.com"
sender_password = "your-app-password"
```

---

## Project Structure

```
your-project/
├── app.py
├── best_crowd_counter_objects.pth    ← YOU NEED THIS
└── .streamlit/
    └── secrets.toml                  ← Optional (for email)
```



