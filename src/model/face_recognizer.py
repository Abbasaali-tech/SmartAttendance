"""
Face Recognition Inference Module for Smart Attendance System
=====================================================================
Author:  Abbas
Description:
    Performs real-time face recognition using the trained LBPH model,
    implementing eye-alignment preprocessing parity, dynamic thresholding,
    calibrated distance-to-similarity confidence translation, and
    confidence-weighted temporal smoothing.
"""

import os
import cv2
import pickle
import time
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import deque


class FaceRecognizer:
    """
    Main face recognition engine that achieves architectural parity with the 
    registration and training phases.

    Resolves:
        1. Non-linear, dynamic calibration of distance scores to similarity percentages.
        2. Loading and applying the dynamically calibrated training threshold.
        3. Elimination of dead mapping variables and clean label decoding.
        4. Confidence-weighted temporal smoothing (majority-vote queue with confidence weight).
        5. Full eye alignment, padding, and illumination normalization parity during inference.
    """

    def __init__(
        self,
        model_path: str = "models/lbph_model.yml",
        labels_path: str = "models/labels.pkl",
        stats_path: str = "models/confidence_stats.pkl",
        face_size: tuple = (200, 200),
        min_detection_confidence: float = 0.6,
        queue_size: int = 8,
        logger_callback=None,
    ):
        """
        Initializes the FaceRecognizer with models and configurations.
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.stats_path = stats_path
        self.face_size = face_size
        self.min_detection_confidence = min_detection_confidence
        self.queue_size = queue_size
        self.logger_callback = logger_callback

        # Models & Metadata placeholders
        self.recognizer = None
        self.label_map = None
        self.confidence_stats = None
        self.dynamic_threshold = 90.0  # safe default fallback

        self.attendance_cooldown_sec = 30
        self.last_attendance_time = {}
        self.last_unknown_time = 0.0
        self.attendance_snapshot_dir = "attendance_snapshots"
        self.unknown_snapshot_dir = "unknown_snapshots"

        # Active prediction queue for confidence-weighted smoothing
        # Holds: (label_id, similarity_percentage)
        self.prediction_queue = deque(maxlen=self.queue_size)

        # Setup MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.min_detection_confidence,
        )

    def _log(self, message: str):
        if self.logger_callback:
            self.logger_callback(message)
        else:
            print(message)

    def load_model_and_metadata(self) -> bool:
        """
        Loads the trained LBPH model weights, label mapping, and confidence stats.
        """
        if not os.path.exists(self.model_path):
            self._log(f"[ERROR] Trained model file not found: '{self.model_path}'.")
            return False

        if not os.path.exists(self.labels_path):
            self._log(f"[ERROR] Label mapping file not found: '{self.labels_path}'.")
            return False

        try:
            # Load LBPH model
            if not hasattr(cv2, "face"):
                self._log("[ERROR] OpenCV contrib is missing. Install opencv-contrib-python to enable cv2.face.")
                return False
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.model_path)

            # Load label dictionary
            with open(self.labels_path, "rb") as f:
                self.label_map = pickle.load(f)

            # Load confidence calibration stats (if available)
            if os.path.exists(self.stats_path):
                with open(self.stats_path, "rb") as f:
                    self.confidence_stats = pickle.load(f)
                
                overall = self.confidence_stats.get("overall", {})
                self.dynamic_threshold = overall.get("dynamic_threshold", self.dynamic_threshold)
                self._log(f"[INFO] Loaded calibrated dynamic threshold: {self.dynamic_threshold:.2f}")
            else:
                self._log(f"[WARNING] Confidence calibration stats not found at '{self.stats_path}'. Using fallback threshold of {self.dynamic_threshold:.1f}.")

            self._log(f"[INFO] Model and metadata loaded successfully! ({len(self.label_map)} registered classes)")
            return True

        except Exception as e:
            self._log(f"[ERROR] Failed to load model metadata: {e}")
            return False

    def preprocess_face(self, frame: np.ndarray, detection, w: int, h: int) -> np.ndarray:
        """
        Performs full preprocessing parity with registration:
        1. Bounding box coordinates with identical padding consistency.
        2. Crop safety check.
        3. Real-time eye alignment to rotate face horizontally.
        4. Grayscale conversion.
        5. Exact uniform resizing.
        6. Histogram equalization.
        """
        # Bbox extraction
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # Exact same 10% padding
        padding_w = int(box_w * 0.1)
        padding_h = int(box_h * 0.1)

        x_start = max(0, x - padding_w)
        y_start = max(0, y - padding_h)
        x_end = min(w, x + box_w + padding_w)
        y_end = min(h, y + box_h + padding_h)

        if not (y_end > y_start and x_end > x_start):
            return None

        # Face crop
        face_crop = frame[y_start:y_end, x_start:x_end]
        if face_crop.size == 0:
            return None

        # Professional Eye Alignment
        if detection.location_data.relative_keypoints:
            kp = detection.location_data.relative_keypoints
            # Keypoint 0: Right Eye (viewer's left), Keypoint 1: Left Eye (viewer's right)
            eye1_x = int(kp[0].x * w) - x_start
            eye1_y = int(kp[0].y * h) - y_start
            eye2_x = int(kp[1].x * w) - x_start
            eye2_y = int(kp[1].y * h) - y_start

            # Sort by X to confirm leftmost and rightmost eyes
            eyes = sorted([(eye1_x, eye1_y), (eye2_x, eye2_y)], key=lambda e: e[0])
            eye_l, eye_r = eyes[0], eyes[1]

            dy = eye_l[1] - eye_r[1]
            dx = eye_r[0] - eye_l[0]

            if dx > 0:
                angle = np.degrees(np.arctan2(dy, dx))
                eye_center = (float(eye_l[0] + eye_r[0]) / 2.0, float(eye_l[1] + eye_r[1]) / 2.0)
                M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                face_crop = cv2.warpAffine(
                    face_crop,
                    M,
                    (face_crop.shape[1], face_crop.shape[0]),
                    flags=cv2.INTER_CUBIC,
                )

        # Convert to Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Exact uniform resize
        gray_resized = cv2.resize(gray, self.face_size)

        # Histogram equalization for illumination parity
        gray_equalized = cv2.equalizeHist(gray_resized)

        return gray_equalized

    def _calculate_similarity(self, conf: float, label_id: int) -> float:
        """
        Translates raw LBPH chi-square distance into a mathematically sound,
        calibrated similarity percentage using training stats.
        """
        if self.confidence_stats is None:
            # Fallback scaling if stats aren't calibrated yet
            return max(0.0, min(100.0, 100.0 - (conf / 1.5)))

        overall = self.confidence_stats.get("overall", {})
        mean_conf = overall.get("mean", 60.0)
        min_conf = overall.get("min", 20.0)
        dyn_thresh = self.dynamic_threshold

        # Optionally use per-class stats if available for hyper-precise tracking
        per_class = self.confidence_stats.get("per_class", {}).get(label_id, {})
        if per_class:
            mean_conf = per_class.get("mean", mean_conf)
            min_conf = per_class.get("min", min_conf)
            dyn_thresh = per_class.get("dynamic_threshold", dyn_thresh)

        # Calibrated non-linear scaling
        if conf <= mean_conf:
            # Very confident: Map to [80%, 100%]
            span = max(1e-5, mean_conf - min_conf)
            ratio = max(0.0, min(1.0, (conf - min_conf) / span))
            return 100.0 - 20.0 * ratio
        elif conf <= dyn_thresh:
            # Acceptable match: Map to [50%, 80%]
            span = max(1e-5, dyn_thresh - mean_conf)
            ratio = max(0.0, min(1.0, (conf - mean_conf) / span))
            return 80.0 - 30.0 * ratio
        else:
            # Unconfident / Reject: Map to [0%, 50%]
            span = max(1e-5, dyn_thresh)
            ratio = max(0.0, (conf - dyn_thresh) / span)
            return max(0.0, 50.0 - 50.0 * ratio)

    def _get_smoothed_prediction(self) -> tuple:
        """
        Applies confidence-weighted temporal smoothing over the sliding prediction queue.
        Sums up similarity scores for each label_id as weights.
        
        Returns:
            (best_label_id, accumulated_confidence_percentage)
        """
        if not self.prediction_queue:
            return -1, 0.0

        weights = {}
        for lbl, score in self.prediction_queue:
            weights[lbl] = weights.get(lbl, 0.0) + score

        # Select predicted label with the highest accumulated weight
        best_label = max(weights, key=weights.get)
        
        # Calculate average similarity score for the best label in the active window
        matching_scores = [score for lbl, score in self.prediction_queue if lbl == best_label]
        avg_score = sum(matching_scores) / len(matching_scores)

        return best_label, avg_score

    def recognize_faces(self, attendance_manager=None, stop_event=None):
        """
        Main execution loop. Opens webcam, processes detections in real-time,
        recognizes identities using LBPH and HUD overlays, and writes predictions.
        If an attendance_manager is provided, marks attendance for confident predictions.
        """
        if not self.load_model_and_metadata():
            self._log("[ERROR] Cannot run recognition without a valid model and label mapping.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log("[ERROR] Could not access webcam. Ensure no other applications are using it.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Style tokens
        font = cv2.FONT_HERSHEY_DUPLEX
        primary_color = (79, 70, 229)    # Slate Indigo (RGB: 229, 70, 79)
        success_color = (34, 197, 94)    # Emerald Green (RGB: 94, 197, 34)
        warning_color = (239, 68, 68)    # Amber Red (RGB: 68, 68, 239)
        text_color = (255, 255, 255)

        self._log("\n" + "=" * 50)
        self._log("          REAL-TIME FACE RECOGNITION (ATTENDANCE)          ")
        self._log("=" * 50)
        self._log("[INFO] Warmup successful. Camera stream opening...")
        self._log("[INFO] Controls:")
        self._log("  - Press 'q' to stop recognition and return to menu.")
        self._log("=" * 50 + "\n")

        os.makedirs(self.attendance_snapshot_dir, exist_ok=True)
        os.makedirs(self.unknown_snapshot_dir, exist_ok=True)

        try:
            while cap.isOpened():
                if stop_event is not None and stop_event.is_set():
                    self._log("[INFO] Attendance stop requested by user.")
                    break
                ret, frame = cap.read()
                if not ret:
                    self._log("[ERROR] Failed to capture video frame.")
                    break

                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape

                display_frame = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.face_detector.process(rgb)
                
                prediction_text = "Unknown"
                pred_color = warning_color
                sim_pct = 0.0

                if results.detections:
                    # Capture largest face for high stability
                    detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width *
                                                                         d.location_data.relative_bounding_box.height)
                    
                    score = detection.score[0] if detection.score else 0.0
                    if score >= self.min_detection_confidence:
                        # Extract coordinates to draw bounding box
                        bbox = detection.location_data.relative_bounding_box
                        x_box = int(bbox.xmin * w)
                        y_box = int(bbox.ymin * h)
                        w_box = int(bbox.width * w)
                        h_box = int(bbox.height * h)
                        
                        # Apply identical padding bounds for the box layout
                        pad_w = int(w_box * 0.1)
                        pad_h = int(h_box * 0.1)
                        x1, y1 = max(0, x_box - pad_w), max(0, y_box - pad_h)
                        x2, y2 = min(w, x_box + w_box + pad_w), min(h, y_box + h_box + pad_h)

                        # Full parity preprocessing
                        preprocessed = self.preprocess_face(frame, detection, w, h)
                        
                        if preprocessed is not None:
                            # Run LBPH predict
                            label_id, distance = self.recognizer.predict(preprocessed)
                            
                            # Calibrated confidence scaling
                            sim_pct = self._calculate_similarity(distance, label_id)
                            
                            # Check dynamically against dataset threshold
                            overall_stats = self.confidence_stats.get("overall", {}) if self.confidence_stats else {}
                            dyn_thresh = overall_stats.get("dynamic_threshold", self.dynamic_threshold)
                            
                            # Classify prediction
                            if distance < dyn_thresh:
                                self.prediction_queue.append((label_id, sim_pct))
                            else:
                                self.prediction_queue.append((-1, 0.0))  # Unknown / Rejected

                            # Apply confidence-weighted temporal smoothing
                            smoothed_label, smoothed_sim = self._get_smoothed_prediction()
                            
                            if smoothed_label != -1 and smoothed_sim >= 50.0:
                                # Retrieve cleanly from label map without dead lookup mappings
                                raw_name = self.label_map.get(smoothed_label, "Unknown")
                                prediction_text = f"{raw_name.replace('_', ' ')} ({smoothed_sim:.1f}%)"
                                pred_color = success_color

                                if attendance_manager is not None and raw_name != "Unknown":
                                    parts = raw_name.rsplit('_', 1)
                                    if len(parts) != 2:
                                        self._log(f"[WARNING] Invalid folder naming format: {raw_name}")
                                        continue
                                    student_name = parts[0] if len(parts) > 1 else raw_name
                                    student_id = parts[1] if len(parts) > 1 else raw_name

                                    now_ts = time.time()
                                    last_ts = self.last_attendance_time.get(student_id, 0.0)
                                    can_mark = now_ts - last_ts >= self.attendance_cooldown_sec
                                    
                                    # Ensure student exists in DB
                                    attendance_manager.add_student(student_id, student_name)
                                    marked = False
                                    if can_mark:
                                        marked = attendance_manager.mark_attendance(student_id, student_name)
                                    if marked:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        snapshot_path = os.path.join(
                                            self.attendance_snapshot_dir,
                                            f"{student_id}_{timestamp}.jpg",
                                        )
                                        cv2.imwrite(snapshot_path, frame)
                                        self.last_attendance_time[student_id] = now_ts
                                        self._log(f"[ATTENDANCE] Marked {raw_name} as Present.")
                                        self._log("[ATTENDANCE] Attendance Marked Successfully")
                                    elif can_mark:
                                        self._log(f"[ATTENDANCE] Duplicate detected for {raw_name}; already marked today.")
                            else:
                                prediction_text = "Unknown / Reject"
                                pred_color = warning_color

                                if attendance_manager is not None:
                                    now_ts = time.time()
                                    if now_ts - self.last_unknown_time >= self.attendance_cooldown_sec:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        snapshot_path = os.path.join(
                                            self.unknown_snapshot_dir,
                                            f"unknown_{timestamp}.jpg",
                                        )
                                        cv2.imwrite(snapshot_path, frame)
                                        attendance_manager.log_unknown(snapshot_path)
                                        self.last_unknown_time = now_ts

                        # Draw bounding box and label
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), pred_color, 2)
                        
                        # Add a small label background tag for the face name
                        tag_size = cv2.getTextSize(prediction_text, font, 0.6, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + tag_size[0] + 10, y1), pred_color, -1)
                        cv2.putText(display_frame, prediction_text, (x1 + 5, y1 - 7), font, 0.6, text_color, 1, cv2.LINE_AA)
                else:
                    self.prediction_queue.clear()

                # --- Premium HUD Overlay ---
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 75), (15, 23, 42), -1)  # Slate dark bg
                cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

                cv2.putText(display_frame, "SMART ATTENDANCE - REAL-TIME INFERENCE ENGINE", (20, 30), 
                            font, 0.7, text_color, 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Active Calibration Threshold: {self.dynamic_threshold:.2f} (Dynamic)", 
                            (20, 53), font, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

                # Instruction bar at the bottom
                cv2.rectangle(display_frame, (0, h - 40), (w, h), (15, 23, 42), -1)
                cv2.putText(display_frame, "Controls: Press [q] to stop recognition and return to main menu", 
                            (20, h - 15), font, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

                cv2.imshow("Face Recognition Attendance System", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._log("[INFO] Recognition interface closed by user.")
                    break

        except Exception as e:
            self._log(f"[ERROR] Recognition loop encountered an issue: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._log("[INFO] Webcam resources released safely.")


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
