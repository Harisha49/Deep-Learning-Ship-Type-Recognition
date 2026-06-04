"""
Marine Collision Avoidance System - Streamlit App
================================================
Real-time computer vision application for detecting ships and 
predicting potential collision risks using YOLO.

Author: Marine Safety AI
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import pygame
import time

# Import utility modules
from utils.detection import VesselDetector
from utils.tracking import CentroidTracker
from utils.collision import CollisionDetector


# Page configuration
st.set_page_config(
    page_title="Marine Collision Avoidance System",
    page_icon="🚢",
    layout="wide"
)


def process_video(video_path, conf_threshold, safety_threshold, model_path, 
                   preview_placeholder=None, status_placeholder=None,
                   pixels_per_meter=50, show_speed=True, 
                   show_detailed_guidance=True, show_distance_eta=True):
    """
    Process video and return annotated frames.
    
    Args:
        video_path: Path to input video
        conf_threshold: YOLO confidence threshold
        safety_threshold: Collision safety distance
        model_path: Path to YOLO model
        preview_placeholder: Streamlit placeholder for video preview
        status_placeholder: Streamlit placeholder for status text
        pixels_per_meter: Conversion factor for speed calculation
        show_speed: Whether to display vessel speed
        show_detailed_guidance: Whether to show detailed navigation guidance
        show_distance_eta: Whether to show distance and ETA
        
    Yields:
        Annotated frames
    """
    # Initialize components
    detector = VesselDetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = CentroidTracker(pixels_per_meter=pixels_per_meter)
    collision_detector = CollisionDetector(safety_threshold=safety_threshold)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Update tracker with actual video FPS
    tracker.set_fps(fps)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect vessels
        detections = detector.detect(frame)
        
        # Track objects
        tracked_objects = tracker.update(detections)
        
        # Check collisions
        collisions = collision_detector.check_collisions(tracked_objects)
        
        # Draw detections
        output_frame = detector.draw_detections(frame, detections)
        
        # Draw tracking IDs
        for obj_id, obj_data in tracked_objects.items():
            cx, cy = obj_data['centroid']
            
            # Draw ID
            cv2.putText(
                output_frame, f"ID:{obj_id}", (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            
            # Draw speed if enabled
            if show_speed:
                speed_knots = obj_data.get('speed_knots', 0)
                speed_text = f"{speed_knots:.1f} kn"
                cv2.putText(
                    output_frame, speed_text, (cx - 25, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
                )
        
        # Draw collision warnings
        output_frame, status = collision_detector.draw_collision_warnings(output_frame, collisions)
        
        # Add info overlay
        if show_speed:
            # Calculate average speed of tracked objects
            avg_speed = np.mean([obj.get('speed_knots', 0) for obj in tracked_objects.values()]) if tracked_objects else 0
            info_text = f"Frame: {frame_count}/{total_frames} | Detected: {len(detections)} | Tracked: {len(tracked_objects)} | Avg Speed: {avg_speed:.1f} kn"
        else:
            info_text = f"Frame: {frame_count}/{total_frames} | Detected: {len(detections)} | Tracked: {len(tracked_objects)}"
        
        cv2.putText(
            output_frame, info_text, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Update preview in real-time if placeholder provided
        if preview_placeholder is not None:
            preview_placeholder.image(output_frame, channels="BGR", caption=f"Live Detection - {status}")
        
        # Update status if placeholder provided
        if status_placeholder is not None:
            status_placeholder.info(f"Processing frame {frame_count}/{total_frames} - Status: {status}")

        yield output_frame, status, tracked_objects, collisions
    
    cap.release()


def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🚢 Marine Collision Avoidance System")
    st.markdown("""
    Real-time vessel detection and collision risk assessment using YOLOv8.
    Upload a marine video to detect ships and predict potential collisions.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model path
    model_path = st.sidebar.text_input(
        "YOLO Model Path",
        value="model/yolov8n.pt",
        help="Path to YOLO model weights"
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence for vessel detection"
    )
    
    # Safety threshold
    safety_threshold = st.sidebar.slider(
        "Safety Threshold (pixels)",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="Distance threshold for collision warning"
    )
    
    # Speed calculation settings
    st.sidebar.header("⚡ Speed Settings")
    
    pixels_per_meter = st.sidebar.slider(
        "Pixels per Meter",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Conversion factor for speed calculation (adjust based on camera view)"
    )
    
    show_speed = st.sidebar.checkbox(
        "Show Vessel Speed",
        value=True,
        help="Display speed of detected vessels in knots"
    )
    
    # Guidance settings
    st.sidebar.header("📋 Guidance Settings")
    
    show_detailed_guidance = st.sidebar.checkbox(
        "Show Detailed Guidance",
        value=True,
        help="Show detailed navigation suggestions during detection"
    )
    
    show_distance_eta = st.sidebar.checkbox(
        "Show Distance & ETA",
        value=True,
        help="Show estimated time to collision and distance"
    )
    
    # Enable live preview option
    live_preview = st.sidebar.checkbox(
        "Enable Live Video Preview",
        value=True,
        help="Show real-time detection preview while processing"
    )

    # Video upload
    st.sidebar.header("📹 Input Video")
    uploaded_file = st.sidebar.file_uploader(
        "Upload MP4 video",
        type=['mp4']
    )
    
    # Check model exists
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        st.info("Please ensure yolov8n.pt is in the model directory")
        return
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        try:
            # Process button
            if st.button("🔄 Process Video", type="primary"):
                # Create placeholders for preview and status
                preview_placeholder = st.empty()
                status_placeholder = st.empty()
                audio_placeholder = st.empty()
                recommendation_placeholder = st.empty()

                # Initialize pygame mixer for audio
                pygame.mixer.init()
                alarm_sound = pygame.mixer.Sound("dangeralarm.mp3")
                warning_sound = pygame.mixer.Sound("wife_warning.mp3")
                alarm_playing = False
                warning_playing = False

                # Create video writer for output
                output_frames = []
                final_status = "SAFE"

                # Process video with live preview
                for frame, status, tracked_objects, collisions in process_video(
                    temp_video_path,
                    conf_threshold,
                    safety_threshold,
                    model_path,
                    preview_placeholder if live_preview else None,
                    status_placeholder if live_preview else None,
                    pixels_per_meter,
                    show_speed,
                    show_detailed_guidance,
                    show_distance_eta
                ):
                    final_status = status
                    output_frames.append(frame)

                    # Get additional info for detailed guidance
                    num_vessels = len(tracked_objects)
                    avg_speed = 0
                    if tracked_objects and show_speed:
                        speeds = [obj.get('speed_knots', 0) for obj in tracked_objects.values()]
                        avg_speed = np.mean(speeds) if speeds else 0
                    
                    # Automatic predictive analysis
                    predictive_info = ""
                    if collisions and show_detailed_guidance:
                        # Analyze the most critical collision
                        worst_collision = min(collisions, key=lambda x: x['distance'])
                        dist = worst_collision['distance']
                        combined = worst_collision.get('combined_speed', 0)
                        
                        # Calculate predicted CPA
                        if combined > 0:
                            time_to_collision = (dist / max(combined, 1)) * 0.1
                            predictive_info = f" | Predicted CPA: {time_to_collision:.1f}s"
                    
                    # Play alarm sound if HIGH risk is detected
                    if status == "HIGH RISK":
                        if not alarm_playing:
                            alarm_sound.play(-1)  # -1 means loop indefinitely
                            alarm_playing = True
                        # Show audio alert in placeholder
                        audio_placeholder.error("🔴 DANGER! COLLISION IMMINENT - ALARM PLAYING!")
                        # Show detailed recommendation
                        if show_detailed_guidance:
                            recommendation_placeholder.warning(
                                """
                                🚨 **IMMEDIATE ACTION REQUIRED:**
                                
                                - **STOP** or reduce speed to minimum immediately!
                                - **SOUND** warning signals (5 short blasts)
                                - **PREPARE** to alter course hard to starboard
                                - **MONITOR** CPA (Closest Point of Approach)
                                - **CONTACT** vessel on VHF Channel 16 if needed
                                """)
                        else:
                            recommendation_placeholder.warning("🚨 IMMEDIATE ACTION REQUIRED: ")
                    elif status == "WARNING":
                        # Stop alarm if it was playing
                        if alarm_playing:
                            alarm_sound.stop()
                            alarm_playing = False
                        # Play warning sound if not already playing
                        if not warning_playing:
                            warning_sound.play(-1)  # Loop indefinitely
                            warning_playing = True
                        
                        if show_detailed_guidance:
                            # Calculate ETA info if enabled
                            eta_info = ""
                            if show_distance_eta and tracked_objects:
                                eta_info = f" | Avg Speed: {avg_speed:.1f} kn"
                            
                            audio_placeholder.warning(
                                f"🟠 WARNING: Vessels approaching dangerously close!{eta_info}")
                            recommendation_placeholder.info(
                                f"""
                                ⚠️ **CAUTION: Prepare for evasive action**
                                
                                - **REDUCE** speed to safe maneuverable speed
                                - **READY** crew for potential maneuver
                                - **MONITOR** the approaching vessel's bearing
                                - **PLAN** alternate course if needed
                                - **CHECK** astern area before maneuvering
                                """)
                        else:
                            audio_placeholder.warning("🟠 WARNING: Vessels approaching dangerously close!")
                            recommendation_placeholder.info("⚠️ CAUTION: Prepare for evasive action - Reduce speed and prepare to alter course!")
                    else:
                        # Stop alarm and warning sounds if risk is SAFE
                        if alarm_playing:
                            alarm_sound.stop()
                            alarm_playing = False
                        if warning_playing:
                            warning_sound.stop()
                            warning_playing = False
                        
                        if show_detailed_guidance:
                            speed_info = f" | Avg Speed: {avg_speed:.1f} kn" if show_speed else ""
                            audio_placeholder.success(f"🟢 SAFE: No collision risk detected{speed_info}")
                            recommendation_placeholder.info(
                                f"""
                                ✅ **Continue normal navigation**
                                
                                - All vessels at safe distance
                                - Monitor for any new contacts
                                - Maintain current speed and course
                                - Stay vigilant for changing conditions
                                """)
                        else:
                            audio_placeholder.success("🟢 SAFE: No collision risk detected")
                            recommendation_placeholder.info("✅ Continue normal navigation - All vessels at safe distance")

                # Stop alarm and warning sounds if they were playing
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False
                if warning_playing:
                    warning_sound.stop()
                    warning_playing = False

                # Clear audio and recommendation placeholders
                audio_placeholder.empty()
                recommendation_placeholder.empty()

                # Create output video
                if output_frames:
                    # Clear preview placeholder
                    if live_preview:
                        preview_placeholder.empty()
                        status_placeholder.empty()
                    
                    # Clear audio placeholder
                    audio_placeholder.empty()
                    recommendation_placeholder.empty()

                    # Save as temporary output
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = output_frames[0].shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

                    for frame in output_frames:
                        out.write(frame)
                    out.release()

                    # Display results
                    st.success(f"✅ Processing complete! Final Status: {final_status}")

                    # Show final frame
                    st.subheader("📷 Final Detection Frame")
                    st.image(output_frames[-1], channels="BGR", caption="Final Detection Result")

                    # Display video
                    st.subheader("🎬 Processed Video")
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()

                    st.video(video_bytes)

                    # Download button
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="processed_marine_video.mp4",
                        mime="video/mp4"
                    )

                    # Cleanup
                    os.remove(output_path)
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    else:
        # Show demo/sample info when no video uploaded
        st.info("👈 Upload a marine video from the sidebar to begin")
        
        # Display sample metrics
        st.subheader("📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", "YOLOv8n")
        col2.metric("Detection Confidence", f"{conf_threshold}")
        col3.metric("Safety Threshold", f"{safety_threshold}px")
        col4.metric("Speed Display", "Enabled" if show_speed else "Disabled")
        
        # Instructions
        st.subheader("📝 Instructions")
        st.markdown("""
        1. **Upload Video**: Select an MP4 marine video from the sidebar
        2. **Adjust Settings**: Configure detection confidence, safety threshold, and speed settings
        3. **Enable Preview**: Check "Enable Live Video Preview" to see real-time detection
        4. **Speed Calculation**: Adjust "Pixels per Meter" based on your camera view for accurate speed
        5. **Detailed Guidance**: Enable "Show Detailed Guidance" for comprehensive navigation suggestions
        6. **Process**: Click "Process Video" to analyze
        7. **Results**: View the annotated output with collision warnings and speed info
        
        ### Detection Legend:
        - 🟢 **Green Box**: Detected vessel
        - 🟡 **Yellow ID**: Unique vessel ID (tracking)
        - 🟠 **Yellow Speed**: Vessel speed in knots
        - 🔴 **Red Line**: High risk collision
        - 🟠 **Orange Line**: Warning collision
        - 🔢 **Combined Speed**: Total speed of colliding vessels
        
        ### Speed Calculation:
        - Speed is calculated based on vessel movement between frames
        - Displayed in knots (nautical miles per hour)
        - Adjust "Pixels per Meter" setting to calibrate for your specific camera setup
        
        ### Live Recommendations:
        - Real-time safety recommendations displayed during detection
        - 🔴 **HIGH RISK**: Immediate action required - `dangeralarm.mp3` plays
        - 🟠 **WARNING**: Prepare for evasive action - `wife_warning.mp3` plays
        - 🟢 **SAFE**: Continue normal navigation
        - Detailed step-by-step guidance available when "Show Detailed Guidance" is enabled
        
        ### Live Preview:
        - Enable "Live Video Preview" to see real-time detection as video processes
        - Shows distance, speed, and estimated time to collision (when enabled)
        """)


if __name__ == "__main__":
    main()
