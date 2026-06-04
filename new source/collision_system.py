"""
Marine Collision Avoidance and Video Enhancement System
========================================================
AI-powered real-time collision avoidance system for marine environments.

Features:
- Live video feed analysis from onboard cameras
- Object detection using YOLOv8 for vessel identification
- Low-light video enhancement using MIRNetv2
- Collision prediction with directional commands
- Visual alerts for safe navigation
"""

import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from runpy import run_path
from ultralytics import YOLO
from torchvision.transforms import ToTensor, ToPILImage


# ==================== CONFIGURATION ====================
# Model paths - using local files in project directory
MODEL_PATHS = {
    'mirnet_weights': 'enhancement_lol.pth',  # Local file
    'yolo_weights': 'yolov8n.pt'  # Local file
}

# Camera settings
CAMERA_INDEX = 0  # 0 for default camera
VIDEO_OUTPUT = 'enhanced_output.mp4'
OUTPUT_FPS = 20.0

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.3
IAA_THRESHOLD = 0.45

# Vessel categories (COCO classes relevant to marine)
VESSEL_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (can be adapted for marine)
ALL_CLASSES = True  # Set to True to detect all objects, False for only vessels


# ==================== MODEL LOADING ====================
def get_weights_and_parameters(task, parameters):
    """Get model weights and parameters for specified task"""
    if task == 'lowlight_enhancement':
        weights = MODEL_PATHS['mirnet_weights']
    return weights, parameters


def load_mirnet_model():
    """Load MIRNetv2 enhancement model"""
    task = 'lowlight_enhancement'
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'n_feat': 80,
        'chan_factor': 1.5,
        'n_RRG': 4,
        'n_MRB': 2,
        'height': 3,
        'width': 2,
        'bias': False,
        'scale': 1,
        'task': task
    }

    weights, parameters = get_weights_and_parameters(task, parameters)
    
    # Load MIRNetv2 architecture
    arch_path = os.path.join('MIRNetv2', 'basicsr', 'models', 'archs', 'mirnet_v2_arch.py')
    load_arch = run_path(arch_path)
    mir_model = load_arch['MIRNet_v2'](**parameters)
    
    # Load weights
    checkpoint = torch.load(weights, map_location='cpu')
    mir_model.load_state_dict(checkpoint['params'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mir_model.to(device).eval()
    
    print(f"[INFO] MIRNetv2 loaded on: {device}")
    return model, device


def load_yolo_model():
    """Load YOLO model for object detection"""
    model = YOLO(MODEL_PATHS['yolo_weights'])
    print(f"[INFO] YOLO model loaded: {MODEL_PATHS['yolo_weights']}")
    return model


# ==================== PREPROCESSING ====================
def preprocess_frame(frame):
    """Preprocess frame for MIRNetv2 (pad to multiples of 8)"""
    h, w, _ = frame.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    padded = cv2.copyMakeBorder(frame, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT)
    return padded


def enhance_frame(model, device, frame, to_tensor, to_pil_image):
    """Enhance low-light frame using MIRNetv2"""
    padded = preprocess_frame(frame)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_tensor = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(img_tensor)
    
    enhanced_img = enhanced.squeeze().detach().cpu()
    enhanced_img = to_pil_image(enhanced_img)
    enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
    
    # Crop back to original size
    enhanced_img = enhanced_img[:frame.shape[0], :frame.shape[1]]
    return enhanced_img


# ==================== COLLISION DETECTION ====================
def calculate_ttc(distance, relative_velocity):
    """Calculate Time To Collision (TTC)"""
    if relative_velocity > 0:
        return distance / relative_velocity
    return float('inf')


def get_maneuver_direction(boxes, frame_width, frame_height):
    """
    Determine maneuver direction based on object positions.
    
    Returns:
        direction: Turn Left, Turn Right, Hold Course, or STOP
        confidence: Confidence score (0-1)
        threat_level: LOW, MEDIUM, HIGH
    """
    if not boxes:
        return "Clear", 1.0, "LOW"
    
    # Analyze object positions
    left_objects = []
    right_objects = []
    center_objects = []
    close_objects = []
    
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    danger_zone_y = frame_height * 0.7  # Bottom 30% of frame is danger zone
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Calculate box center
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        box_width = x2 - x1
        
        # Check if in danger zone (close to camera/bottom of frame)
        if y2 > danger_zone_y:
            close_objects.append({
                'x': box_center_x,
                'y': box_center_y,
                'width': box_width,
                'y2': y2
            })
        
        # Categorize by position
        if box_center_x < frame_center_x - frame_width * 0.1:
            left_objects.append(box_center_x)
        elif box_center_x > frame_center_x + frame_width * 0.1:
            right_objects.append(box_center_x)
        else:
            center_objects.append(box_center_x)
    
    # Determine threat level and action
    n_close = len(close_objects)
    n_left = len(left_objects)
    n_right = len(right_objects)
    n_center = len(center_objects)
    total = n_close + n_left + n_right + n_center
    
    if total == 0:
        return "Clear", 1.0, "LOW"
    
    # HIGH threat: objects in danger zone
    if n_close > 0:
        # Determine if should go left or right
        close_left = sum(1 for obj in close_objects if obj['x'] < frame_center_x)
        close_right = n_close - close_left
        
        if close_left > close_right:
            direction = "Turn Right"
            confidence = close_left / n_close
        else:
            direction = "Turn Left"
            confidence = close_right / n_close
        
        # Check for head-on collision (center objects in danger zone)
        center_close = [obj for obj in close_objects if abs(obj['x'] - frame_center_x) < frame_width * 0.1]
        if center_close:
            direction = "STOP"
            confidence = 0.95
        
        return direction, min(0.99, confidence), "HIGH"
    
    # MEDIUM threat: objects approaching from sides
    if n_left > n_right:
        return "Turn Right", min(0.85, n_left / total), "MEDIUM"
    elif n_right > n_left:
        return "Turn Left", min(0.85, n_right / total), "MEDIUM"
    
    # LOW threat: objects far away or none
    return "Hold Course", 0.9, "LOW"


def draw_collision_warning(img, direction, confidence, threat_level):
    """Draw collision warning on frame"""
    h, w = img.shape[:2]
    
    # Color based on threat level
    if threat_level == "HIGH":
        color = (0, 0, 255)  # Red
        alert_text = "⚠️ DANGER"
    elif threat_level == "MEDIUM":
        color = (0, 165, 255)  # Orange
        alert_text = "⚡ WARNING"
    else:
        color = (0, 255, 0)  # Green
        alert_text = "✓ CLEAR"
    
    # Draw threat level indicator
    cv2.rectangle(img, (10, 10), (w - 10, 80), color, 2)
    
    # Draw alert text
    cv2.putText(img, alert_text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # Draw direction command
    cmd_text = f"{direction} ({confidence:.0%})"
    cv2.putText(img, cmd_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return img


# ==================== MAIN LOOP ====================
def main():
    """Main function to run the collision avoidance system"""
    print("=" * 60)
    print("Marine Collision Avoidance & Video Enhancement System")
    print("=" * 60)
    
    # Initialize models
    print("\n[1/4] Loading MIRNetv2 enhancement model...")
    mirnet_model, device = load_mirnet_model()
    
    print("\n[2/4] Loading YOLO detection model...")
    yolo_model = load_yolo_model()
    
    # Initialize video capture
    print(f"\n[3/4] Initializing camera (index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"[INFO] Video resolution: {frame_width}x{frame_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, OUTPUT_FPS, (frame_width, frame_height))
    
    # Initialize transforms
    to_tensor = ToTensor()
    to_pil_image = ToPILImage()
    
    print("\n[4/4] Starting detection loop...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame read failed!")
            break
        
        start_time = time.time()
        frame_count += 1
        
        # YOLO Detection
        results = yolo_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        boxes = results.boxes
        
        # Filter for vessel classes if needed
        if not ALL_CLASSES:
            boxes = [box for box in boxes if int(box.cls[0]) in VESSEL_CLASSES]
        
        # Frame Enhancement
        enhanced_frame = enhance_frame(mirnet_model, device, frame, to_tensor, to_pil_image)
        
        # Draw detection boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = f"{yolo_model.names[cls]} {conf:.2f}"
            
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(enhanced_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Get maneuver direction
        direction, confidence, threat_level = get_maneuver_direction(
            boxes, frame_width, frame_height
        )
        
        # Draw collision warning
        enhanced_frame = draw_collision_warning(enhanced_frame, direction, confidence, threat_level)
        
        # FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(enhanced_frame, f"FPS: {fps:.2f}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Marine Collision Avoidance System', enhanced_frame)
        
        # Write to output video
        out.write(enhanced_frame)
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            print(f"[Frame {frame_count}] {direction} | FPS: {fps:.2f}")
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] User requested exit")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Output saved to: {VIDEO_OUTPUT}")
    print(f"[INFO] Total frames processed: {frame_count}")
    print(f"[INFO] Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
