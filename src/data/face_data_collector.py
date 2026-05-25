"""
Face Registration Module for Smart Attendance System
Author: Abbas
Description: Captures face images from a webcam, detects faces using MediaPipe,
             crops the face region, and saves them to a dataset directory.
"""

import os
import cv2
import mediapipe as mp
import time
import numpy as np

class FaceDataCollector:
    """
    A class to handle face data collection for registering new students in the 
    Smart Attendance System. It captures video frames from the webcam, detects 
    faces using MediaPipe, draws bounding boxes, and crops & saves 50 face samples.
    """
    
    def __init__(self, dataset_path="dataset", max_samples=50, min_detection_confidence=0.6,
                 student_name: str = None, student_id: str = None):
        """
        Initializes the FaceDataCollector with configuration parameters.
        
        Args:
            dataset_path (str): Base directory where student face datasets will be stored.
            max_samples (int): Total number of face image samples to save per student.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
        """
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.min_detection_confidence = min_detection_confidence
        self.student_name = student_name
        self.student_id = student_id
        
        # Create dataset base directory if it doesn't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Initialize MediaPipe Face Detection solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Setup the face detector
        # model_selection=0: short-range (best for webcams within 2 meters)
        # model_selection=1: full-range (best for cameras within 5 meters)
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.min_detection_confidence
        )

    def _get_largest_face(self, detections):
        """
        Helper method to select the largest face based on bounding box width * height.
        This provides high tracking stability if background or multiple faces are detected.
        """
        return max(detections, key=lambda d: d.location_data.relative_bounding_box.width *
                                             d.location_data.relative_bounding_box.height)

    def get_student_info(self):
        """
        Interactively prompts the user for student details (Name and ID)
        and prepares the directory structure.
        
        Returns:
            tuple: (student_name, student_id, student_dir)
        """
        print("\n" + "="*50)
        print("          STUDENT FACE REGISTRATION MODULE          ")
        print("="*50)
        
        student_name = (self.student_name or "").strip().replace(" ", "_")
        student_id = (self.student_id or "").strip()

        while True:
            if student_name:
                break
            student_name = input("Enter Student Full Name: ").strip().replace(" ", "_")
            if not student_name:
                print("Error: Student name cannot be empty. Please try again.")

        while True:
            if student_id:
                break
            student_id = input("Enter Student ID (alphanumeric/numbers): ").strip()
            if not student_id:
                print("Error: Student ID cannot be empty. Please try again.")
            
        # Define and create the unique student directory
        student_dir = os.path.join(self.dataset_path, f"{student_name}_{student_id}")
        os.makedirs(student_dir, exist_ok=True)
        
        print(f"\n[INFO] Saving images to: {student_dir}")
        print("[INFO] Controls:")
        print("  - Press 's' to START / PAUSE saving face images.")
        print("  - Press 'q' to QUIT the application at any time.")
        print("="*50 + "\n")
        
        return student_name, student_id, student_dir

    def collect_faces(self):
        """
        Opens the webcam, processes the feed, detects faces using MediaPipe, 
        and crops & saves face samples when triggered.
        """
        # Step 1: Prompt for student details
        student_name, student_id, student_dir = self.get_student_info()
        
        # Step 2: Initialize webcam capture (0 is usually the default built-in camera)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not access the webcam. Please ensure it is connected and not in use by another program.")
            return
            
        # Configure camera resolution for high-quality captures
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # State variables
        sample_count = 0
        is_saving = False
        last_save_time = 0
        save_delay = 0.15  # Seconds to wait between saving each image (avoid duplicate frame spikes)
        
        # Keep a history of the last saved crop to reject duplicate static frames
        prev_face_crop = None
        show_static_warning = False
        
        # Setup aesthetic fonts and colors
        font = cv2.FONT_HERSHEY_DUPLEX
        primary_color = (79, 70, 229)    # Elegant Indigo (BGR: RGB(229, 70, 79))
        accent_color = (34, 197, 94)     # Successful Green (BGR: RGB(94, 197, 34))
        warning_color = (239, 68, 68)    # Vibrant Red (BGR: RGB(68, 68, 239))
        text_color = (255, 255, 255)     # Pure White
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from webcam.")
                    break
                
                # Flip the frame horizontally to behave like a natural mirror
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                # Copy frame for drawing bounding boxes and labels
                display_frame = frame.copy()
                
                # MediaPipe requires RGB images, OpenCV captures in BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Perform Face Detection
                results = self.face_detector.process(rgb_frame)
                
                face_detected = False
                
                # If one or more faces are detected
                if results.detections:
                    # Pick the largest face (increases tracking stability)
                    detection = self._get_largest_face(results.detections)
                    
                    # Verify confidence threshold score explicitly
                    score = detection.score[0] if detection.score else 0.0
                    if score >= self.min_detection_confidence:
                        face_detected = True
                        
                        # Extract bounding box normalized coordinates
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert normalized coordinates to absolute pixel coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        box_w = int(bbox.width * w)
                        box_h = int(bbox.height * h)
                        
                        # Apply a safety margin (padding) around the face crop for better recognition features
                        padding_w = int(box_w * 0.1)
                        padding_h = int(box_h * 0.1)
                        
                        # Mathematically restrict coordinate bounds to prevent IndexError crashes
                        x_start = max(0, x - padding_w)
                        y_start = max(0, y - padding_h)
                        x_end = min(w, x + box_w + padding_w)
                        y_end = min(h, y + box_h + padding_h)
                        
                        # Crop safety check (ensure valid width and height)
                        if y_end > y_start and x_end > x_start:
                            # Draw a premium rounded bounding box on the display frame
                            cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), 
                                          accent_color if is_saving else primary_color, 2)
                            
                            # Draw a small indicator circle on key face points (e.g., nose)
                            if detection.location_data.relative_keypoints:
                                nose = detection.location_data.relative_keypoints[2]
                                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                                cv2.circle(display_frame, (nose_x, nose_y), 4, accent_color, -1)
                            
                            # Handle face image saving
                            if is_saving and (time.time() - last_save_time >= save_delay):
                                face_crop = frame[y_start:y_end, x_start:x_end]
                                
                                # Verify the cropped image is not empty
                                if face_crop.size > 0:
                                    # --- Professional Eye Alignment Preprocessing ---
                                    if detection.location_data.relative_keypoints:
                                        kp = detection.location_data.relative_keypoints
                                        # Keypoint 0: Right Eye (viewer's left), Keypoint 1: Left Eye (viewer's right)
                                        eye1_x = int(kp[0].x * w) - x_start
                                        eye1_y = int(kp[0].y * h) - y_start
                                        eye2_x = int(kp[1].x * w) - x_start
                                        eye2_y = int(kp[1].y * h) - y_start
                                        
                                        # Sort eyes by X-coordinate to guarantee left-to-right eye order in image space
                                        eyes = sorted([(eye1_x, eye1_y), (eye2_x, eye2_y)], key=lambda e: e[0])
                                        eye_l, eye_r = eyes[0], eyes[1]
                                        
                                        # Calculate eye tilt angle
                                        # dy is positive if left eye is lower in the image than right eye
                                        dy = eye_l[1] - eye_r[1]
                                        dx = eye_r[0] - eye_l[0]
                                        
                                        if dx > 0:
                                            angle = np.degrees(np.arctan2(dy, dx))
                                            # Center of rotation is the midpoint between both eyes
                                            eye_center = (float(eye_l[0] + eye_r[0]) / 2.0, float(eye_l[1] + eye_r[1]) / 2.0)
                                            # Create rotation matrix and warp the face crop
                                            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                                            face_crop = cv2.warpAffine(
                                                face_crop,
                                                M,
                                                (face_crop.shape[1], face_crop.shape[0]),
                                                flags=cv2.INTER_CUBIC
                                            )
                                    
                                    # Dynamic frame difference filter
                                    is_duplicate = False
                                    if prev_face_crop is not None:
                                        # Convert crops to grayscale and resize to a static 100x100
                                        gray_curr = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                                        gray_prev = cv2.cvtColor(prev_face_crop, cv2.COLOR_BGR2GRAY)
                                        
                                        resized_curr = cv2.resize(gray_curr, (100, 100))
                                        resized_prev = cv2.resize(gray_prev, (100, 100))
                                        
                                        # Calculate Mean Absolute Error (MAE)
                                        abs_diff = cv2.absdiff(resized_curr, resized_prev)
                                        mae = abs_diff.mean()
                                        
                                        # Threshold 6.0: frames below this are identical duplicates (user is still)
                                        if mae < 6.0:
                                            is_duplicate = True
                                    
                                    if not is_duplicate:
                                        sample_count += 1
                                        img_filename = os.path.join(student_dir, f"face_{sample_count}.jpg")
                                        
                                        # Save high quality JPEG
                                        cv2.imwrite(img_filename, face_crop)
                                        last_save_time = time.time()
                                        
                                        # Save a deep copy of the crop for next frame comparison
                                        prev_face_crop = face_crop.copy()
                                        show_static_warning = False
                                        
                                        # Log progress to console
                                        print(f"[INFO] Saved {sample_count}/{self.max_samples}: {img_filename}")
                                        
                                        if sample_count >= self.max_samples:
                                            print(f"\n[SUCCESS] Successfully registered {student_name}! Saved {self.max_samples} images.")
                                            break
                                    else:
                                        show_static_warning = True
                        else:
                            # Coordinates were degenerate/invalid
                            face_detected = False
                
                # --- Drawing HUD (Heads-up Display) Overlay ---
                # Semi-transparent background panel for text readability
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (15, 23, 42), -1)  # Sleek slate dark background
                cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
                
                # App Title & Branding
                cv2.putText(display_frame, "SMART ATTENDANCE - FACE REGISTRATION", (20, 30), 
                            font, 0.7, text_color, 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Student: {student_name.replace('_', ' ')} (ID: {student_id})", 
                            (20, 55), font, 0.5, (203, 213, 225), 1, cv2.LINE_AA)
                
                # Status Indicators
                if is_saving:
                    # Pulsing green indicator for active capture state
                    status_text = f"SAVING: {sample_count}/{self.max_samples} (Tilt head slowly)"
                    color = accent_color
                else:
                    status_text = "READY - Press 's' to start capture"
                    color = primary_color
                    
                cv2.putText(display_frame, status_text, (w - 480, 35), font, 0.6, color, 2, cv2.LINE_AA)
                
                # Instruction Bar at the bottom
                cv2.rectangle(display_frame, (0, h - 40), (w, h), (15, 23, 42), -1)
                cv2.putText(display_frame, "Controls: [s] Start/Pause Saving | [q] Cancel & Quit Registration", 
                            (20, h - 15), font, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
                
                # Alert warning if no face is detected or if confidence is low
                if not face_detected:
                    cv2.putText(display_frame, "WARNING: NO FACE DETECTED", 
                                (w // 2 - 180, h // 2), font, 0.7, warning_color, 2, cv2.LINE_AA)
                elif show_static_warning and is_saving:
                    # Prompt the user to move slightly to introduce diversity in registration
                    cv2.putText(display_frame, "STATIC FRAME - PLEASE MOVE SLIGHTLY", 
                                (w // 2 - 260, h // 2), font, 0.7, warning_color, 2, cv2.LINE_AA)
                
                # Display the finalized frame
                cv2.imshow("Face Registration Module", display_frame)
                
                # Key press listener (wait 1ms)
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' key to quit
                if key == ord('q'):
                    print("[INFO] Registration cancelled by the user.")
                    break
                # 's' key to toggle saving
                elif key == ord('s'):
                    if face_detected:
                        is_saving = not is_saving
                        print(f"[INFO] Image capture status: {'STARTED' if is_saving else 'PAUSED'}")
                    else:
                        print("[WARNING] Cannot start saving: No face detected in frame.")
                        
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            
        finally:
            # Clean release of webcam resource and closing windows
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Webcam released and windows closed.")

if __name__ == "__main__":
    # Self-contained testing block
    collector = FaceDataCollector()
    collector.collect_faces()
