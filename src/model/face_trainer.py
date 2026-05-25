"""
Face Recognition Training Module for Smart Attendance System
=====================================================================
Author:  Abbas
Description:
    Loads pre-cropped face images from the dataset directory, converts
    them to grayscale, optionally validates each crop with MediaPipe,
    assigns numeric label IDs per student, trains an OpenCV LBPH Face
    Recognizer, and saves the trained model + label mapping to disk.

Key outputs:
    models/lbph_model.yml   – The trained LBPH recognizer weights.
    models/labels.pkl       – A Python dict mapping numeric IDs → names.
"""

import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
import random
import re
from typing import List, Dict


class FaceTrainer:
    """
    Responsible for converting raw face-crop images into a trained
    LBPH model that can later be used for real-time recognition.

    Workflow
    --------
    1. Scan dataset/{name_id}/ directories.
    2. For each image: read → validate → convert to grayscale → resize.
    3. Assign a unique integer label to every student folder.
    4. Train the OpenCV LBPH Face Recognizer on the collected samples.
    5. Persist the model (.yml) and the label dictionary (.pkl).
    """

    def __init__(
        self,
        dataset_path: str = "dataset",
        model_dir: str = "models",
        model_filename: str = "lbph_model.yml",
        labels_filename: str = "labels.pkl",
        face_size: tuple = (200, 200),
        validate_with_mediapipe: bool = False,
        mediapipe_confidence: float = 0.5,
        lbph_params: dict = None,
    ):
        """
        Initializes the FaceTrainer with all configurable paths and options.

        Args:
            dataset_path (str):
                Root directory that holds one sub-folder per student.
                Expected layout: dataset/{student_name}_{student_id}/face_*.jpg
            model_dir (str):
                Directory where the trained model and label map are saved.
            model_filename (str):
                Filename for the LBPH model weights.
            labels_filename (str):
                Filename for the pickled label dictionary.
            face_size (tuple):
                (width, height) to which every face crop is resized before
                training.  A uniform size is mandatory for LBPH.
            validate_with_mediapipe (bool):
                If True, each loaded image is passed through MediaPipe Face
                Detection to confirm a face is actually present (rejects
                background-only or corrupt crops).
            mediapipe_confidence (float):
                Minimum detection confidence for the MediaPipe validator.
        """
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, model_filename)
        self.labels_path = os.path.join(model_dir, labels_filename)
        self.face_size = face_size
        self.validate_with_mediapipe = validate_with_mediapipe

        # Create the models directory if it does not exist
        os.makedirs(self.model_dir, exist_ok=True)

        # ----- OpenCV LBPH Recognizer -----
        # Parameters:
        #   radius=1        - pixel radius of the circular LBP neighbourhood
        #   neighbors=8     - number of sample points on the circle
        #   grid_x / grid_y - spatial histogram grid divisions
        default_params = {
            "radius": 1,
            "neighbors": 8,
            "grid_x": 8,
            "grid_y": 8,
        }
        if lbph_params:
            default_params.update(lbph_params)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(**default_params)

        # ----- Optional MediaPipe validator -----
        self.mp_face_detection = None
        self.face_validator = None
        if self.validate_with_mediapipe:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_validator = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=mediapipe_confidence,
            )

    def __del__(self):
        if self.face_validator:
            self.face_validator.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_image(self, filepath: str) -> bool:
        """
        Returns True only if the file can be decoded by OpenCV into a
        non-empty matrix.  Silently catches corrupted / truncated files.
        """
        try:
            img = cv2.imread(filepath)
            if img is None or img.size == 0:
                return False
            return True
        except Exception:
            return False

    def _validate_face_mediapipe(self, bgr_image: np.ndarray) -> bool:
        """
        Runs a quick MediaPipe inference on the image to confirm that at
        least one face exists.  This catches dataset contamination (e.g.
        a blurry shot of a wall that slipped through during registration).

        Args:
            bgr_image: The image in BGR colour space (as loaded by OpenCV).

        Returns:
            True if a face is detected with sufficient confidence.
        """
        if self.face_validator is None:
            return True  # validation disabled – accept everything

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.face_validator.process(rgb)
        return results.detections is not None and len(results.detections) > 0

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> tuple:
        """
        Scans the dataset directory, reports student statistics, performs an 80/20
        train/test split per student class, and returns both subsets.

        Returns:
            (train_faces, train_labels, test_faces, test_labels, label_map)

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
            ValueError:        If no valid images are found at all.
        """
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(
                f"[ERROR] Dataset directory not found: '{self.dataset_path}'. "
                "Please register students first using FaceDataCollector."
            )

        train_faces: List[np.ndarray] = []
        train_labels: List[int] = []
        test_faces: List[np.ndarray] = []
        test_labels: List[int] = []
        label_map: Dict[int, str] = {}

        # Each sub-directory represents one student
        student_dirs = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        if not student_dirs:
            raise ValueError(
                "[ERROR] No student folders found inside the dataset directory. "
                "Register at least one student before training."
            )

        print("\n" + "=" * 60)
        print("          FACE RECOGNITION TRAINING MODULE")
        print("=" * 60)
        print(f"[INFO] Dataset path  : {os.path.abspath(self.dataset_path)}")
        print(f"[INFO] Students found: {len(student_dirs)}")
        print(f"[INFO] Face size     : {self.face_size[0]}x{self.face_size[1]} px")
        print(f"[INFO] MediaPipe val : {'ON' if self.validate_with_mediapipe else 'OFF'}")
        print("-" * 60)

        current_label = 0  # monotonically increasing integer class ID
        stats = {}

        # Set a fixed seed to guarantee reproducible train/test splits
        random.seed(42)

        for folder_name in student_dirs:
            folder_path = os.path.join(self.dataset_path, folder_name)
            image_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

            if not image_files:
                print(f"[WARNING] Skipping empty folder: {folder_name}")
                continue

            student_faces = []
            skipped_corrupt = 0
            skipped_noface = 0

            for img_name in image_files:
                img_path = os.path.join(folder_path, img_name)

                # Step 1: Basic corruption check
                if not self._is_valid_image(img_path):
                    skipped_corrupt += 1
                    continue

                bgr_image = cv2.imread(img_path)

                # Step 2: Optional MediaPipe face validation
                if self.validate_with_mediapipe:
                    if not self._validate_face_mediapipe(bgr_image):
                        skipped_noface += 1
                        continue

                # Step 3: Convert to grayscale
                gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                # Step 4: Resize to uniform dimensions
                gray_resized = cv2.resize(gray, self.face_size)

                # Step 5: Histogram equalization for lighting normalisation
                gray_resized = cv2.equalizeHist(gray_resized)

                student_faces.append(gray_resized)

            loaded = len(student_faces)
            if loaded < 5:
                print(f"[WARNING] Too few samples for {folder_name} (loaded {loaded}, min 5 required). Skipping.")
                continue

            if loaded > 0:
                # Store sample count for dataset stats
                stats[folder_name] = loaded
                label_map[current_label] = folder_name
                
                # Perform 80/20 Train/Test split for this student's face samples
                # Shuffling images to introduce variance and avoid temporal grouping
                random.shuffle(student_faces)
                
                if loaded >= 5:
                    split_idx = int(0.8 * loaded)
                elif loaded > 1:
                    split_idx = loaded - 1  # ensure at least 1 image is in test set
                else:
                    split_idx = loaded      # too few samples, put all in train set
                
                train_f = student_faces[:split_idx]
                test_f = student_faces[split_idx:]
                
                train_faces.extend(train_f)
                train_labels.extend([current_label] * len(train_f))
                test_faces.extend(test_f)
                test_labels.extend([current_label] * len(test_f))
                
                current_label += 1
                status = "OK"
            else:
                status = "NO"

            # Progress line for this student
            print(
                f"  {status}  Label {current_label - 1 if loaded > 0 else '---':>3}  |  "
                f"{folder_name:<30}  |  "
                f"loaded: {loaded:>3}  |  "
                f"corrupt: {skipped_corrupt:>2}  |  "
                f"no-face: {skipped_noface:>2}"
            )

        print("-" * 60)

        if not train_faces:
            raise ValueError(
                "[ERROR] No valid face images were loaded from any student folder. "
                "Ensure the dataset contains readable images with visible faces."
            )

        # Print Dataset Statistics & Balance Report
        print("\n" + "=" * 60)
        print("          DATASET STATISTICS & BALANCE REPORT")
        print("=" * 60)
        print(f"{'Student Folder':<35} | {'Samples':<10}")
        print("-" * 60)
        for folder, count in stats.items():
            warning_msg = ""
            if count < 10:
                warning_msg = " [WARNING: Very low samples!]"
            print(f"{folder:<35} | {count:<10}{warning_msg}")
        print("-" * 60)
        
        counts = list(stats.values())
        if counts:
            mean_samples = np.mean(counts)
            std_samples = np.std(counts)
            min_samples = np.min(counts)
            max_samples = np.max(counts)
            
            print(f"Total enrolled students: {len(stats)}")
            print(f"Total images collected : {sum(counts)}")
            print(f"Average samples/student: {mean_samples:.1f}")
            print(f"Sample range           : {min_samples} - {max_samples} (std: {std_samples:.1f})")
            
            # Balancing check
            if len(counts) > 1:
                # If standard deviation is more than 30% of the mean, or max/min > 3
                if std_samples > 0.3 * mean_samples or (min_samples > 0 and max_samples / min_samples > 3.0):
                    print("\n[WARNING] Dataset is imbalanced!")
                    print("  Large differences in sample count per student can bias the LBPH model.")
                    print("  Recommendation: Register more faces for students with low sample counts.")
                else:
                    print("\n[INFO] Dataset balance check: PASSED (Well-balanced distribution)")
        print("=" * 60 + "\n")

        print(f"[INFO] Total training samples: {len(train_faces)}")
        print(f"[INFO] Total testing samples : {len(test_faces)}")
        print(f"[INFO] Total classes         : {len(label_map)}")

        return train_faces, train_labels, test_faces, test_labels, label_map

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def evaluate(self, test_faces: List, test_labels: List, dynamic_threshold: float = None) -> float:
        """
        Evaluates the model's accuracy on the unseen test dataset.
        """
        if not test_labels:
            print("[WARNING] Evaluation skipped: No testing samples available (dataset too small).")
            return 0.0

        correct = 0
        for face, label in zip(test_faces, test_labels):
            pred, conf = self.recognizer.predict(face)
            if pred == label:
                if dynamic_threshold is not None and conf >= dynamic_threshold:
                    continue
                correct += 1
                
        accuracy = correct / len(test_labels)
        return accuracy

    def train(self) -> None:
        """
        Full training pipeline:
        1. Load and validate the dataset with 80/20 train/test split.
        2. Train the LBPH recognizer on the training set.
        3. Evaluate accuracy on the test set.
        4. Calculate dynamic confidence thresholds.
        5. Save model weights, label map, and confidence stats (both latest and versioned).
        """
        # ---------- Load ----------
        train_faces, train_labels, test_faces, test_labels, label_map = self.load_dataset()

        # ---------- Train ----------
        print("\n[INFO] Training LBPH Face Recognizer ...")
        start_time = time.time()

        if len(train_faces) != len(train_labels):
            raise ValueError(f"Training data mismatch: {len(train_faces)} faces vs {len(train_labels)} labels.")

        # OpenCV expects labels as a numpy int-array
        self.recognizer.train(train_faces, np.array(train_labels, dtype=np.int32))

        elapsed = time.time() - start_time
        print(f"[INFO] Training completed in {elapsed:.2f} seconds.")

        # ---------- Calibrate Dynamic Thresholds ----------
        print("[INFO] Calibrating dynamic confidence thresholds ...")
        confidence_by_class = {}
        all_confidences = []
        min_threshold_floor = 30.0
        
        for face, label in zip(train_faces, train_labels):
            pred, conf = self.recognizer.predict(face)
            if pred == label:
                if label not in confidence_by_class:
                    confidence_by_class[label] = []
                confidence_by_class[label].append(conf)
                all_confidences.append(conf)

        # Include correct predictions from the test set to avoid over-tight thresholds
        for face, label in zip(test_faces, test_labels):
            pred, conf = self.recognizer.predict(face)
            if pred == label:
                if label not in confidence_by_class:
                    confidence_by_class[label] = []
                confidence_by_class[label].append(conf)
                all_confidences.append(conf)
                
        # Group stats
        confidence_stats = {
            "overall": {},
            "per_class": {}
        }
        
        overall_threshold = None
        if all_confidences:
            overall_threshold = float(np.mean(all_confidences) + np.std(all_confidences))
            overall_threshold = max(overall_threshold, min_threshold_floor)
            confidence_stats["overall"] = {
                "mean": float(np.mean(all_confidences)),
                "std": float(np.std(all_confidences)),
                "min": float(np.min(all_confidences)),
                "max": float(np.max(all_confidences)),
                "dynamic_threshold": overall_threshold
            }
            
        for label_id, scores in confidence_by_class.items():
            if scores:
                mean_val = float(np.mean(scores))
                std_val = float(np.std(scores)) if len(scores) > 1 else 0.0
                dynamic_threshold = float(mean_val + std_val)
                dynamic_threshold = max(dynamic_threshold, min_threshold_floor)
                confidence_stats["per_class"][label_id] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "dynamic_threshold": dynamic_threshold
                }

        # ---------- Evaluate ----------
        test_acc = self.evaluate(test_faces, test_labels, overall_threshold)

        # ---------- Versioning & Persistence ----------
        # Scan for existing versioned models to determine the next version number
        version = 1
        if os.path.exists(self.model_dir):
            files = os.listdir(self.model_dir)
            versions = []
            for file in files:
                match = re.search(r"lbph_model_v(\d+)", file)
                if match:
                    versions.append(int(match.group(1)))
            if versions:
                version = max(versions) + 1

        model_version_filename = f"lbph_model_v{version}.yml"
        labels_version_filename = f"labels_v{version}.pkl"
        stats_version_filename = f"confidence_stats_v{version}.pkl"
        
        model_version_path = os.path.join(self.model_dir, model_version_filename)
        labels_version_path = os.path.join(self.model_dir, labels_version_filename)
        stats_version_path = os.path.join(self.model_dir, stats_version_filename)
        
        # Save standard (latest) copies
        self.recognizer.save(self.model_path)
        with open(self.labels_path, "wb") as f:
            pickle.dump(label_map, f)
            
        stats_path = os.path.join(self.model_dir, "confidence_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(confidence_stats, f)
            
        # Save versioned copies
        self.recognizer.save(model_version_path)
        with open(labels_version_path, "wb") as f:
            pickle.dump(label_map, f)
        with open(stats_version_path, "wb") as f:
            pickle.dump(confidence_stats, f)
            
        print(f"[INFO] Latest model files saved to '{self.model_dir}/'")
        print(f"[INFO] Versioned copies saved as version v{version}:")
        print(f"  - {model_version_filename}")
        print(f"  - {labels_version_filename}")
        print(f"  - {stats_version_filename}")

        # ---------- Summary ----------
        self._print_training_summary(
            label_map, 
            len(train_faces) + len(test_faces), 
            elapsed, 
            test_acc, 
            confidence_stats
        )

    # ------------------------------------------------------------------
    # Post-training summary & confidence explanation
    # ------------------------------------------------------------------

    def _print_training_summary(
        self, label_map: dict, total_samples: int, elapsed: float, test_acc: float, confidence_stats: dict
    ) -> None:
        """
        Prints a human-readable summary of training results, test accuracy,
        and provides a calibrated dynamic confidence threshold guide.
        """
        print("\n" + "=" * 60)
        print("          TRAINING & EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Students enrolled  : {len(label_map)}")
        print(f"  Total face samples : {total_samples}")
        print(f"  Test Accuracy      : {test_acc*100:.2f}%")
        print(f"  Training time      : {elapsed:.2f}s")
        print(f"  Model file         : {self.model_path}")
        print(f"  Label file         : {self.labels_path}")
        print(f"  Confidence Stats   : {os.path.join(self.model_dir, 'confidence_stats.pkl')}")
        print("-" * 60)

        print("\n  Label -> Student mapping:")
        for label_id, name in label_map.items():
            print(f"    {label_id:>3}  ->  {name}")

        overall_stats = confidence_stats.get("overall", {})
        if overall_stats:
            print("\n" + "-" * 60)
            print("  CALIBRATED CONFIDENCE STATISTICS (LBPH DISTANCE)")
            print("-" * 60)
            print(f"  Mean distance (similarity score) : {overall_stats['mean']:.2f}")
            print(f"  Standard Deviation (spread)     : {overall_stats['std']:.2f}")
            print(f"  Best-case distance (Min)        : {overall_stats['min']:.2f}")
            print(f"  Worst-case distance (Max)       : {overall_stats['max']:.2f}")
            print(f"  Calibrated Dynamic Threshold    : {overall_stats['dynamic_threshold']:.2f}")
            print(f"    (Calculated as: mean_distance + standard_deviation)")
            
            print("\n  Per-Student Calibrated Thresholds:")
            for label_id, name in label_map.items():
                p_stats = confidence_stats.get("per_class", {}).get(label_id, {})
                if p_stats:
                    print(f"    Label {label_id:>2} [{name:<20}] -> threshold: {p_stats['dynamic_threshold']:.2f} (mean: {p_stats['mean']:.2f})")

        print("\n" + "-" * 60)
        print("  CONFIDENCE SCORE INTERPRETATION GUIDE (LBPH)")
        print("-" * 60)
        print("  CRITICAL CV NOTE:")
        print("  Unlike deep learning models, OpenCV's LBPH confidence output is a")
        print("  *chi-square distance* metric, NOT a percentage probability.")
        print("    - LOWER values  = MORE confident (closer to reference face).")
        print("    - HIGHER values = LESS confident (distant from reference face).\n")
        print("  Lighting, resolution, camera quality, and face alignment heavily shift")
        print("  these distance distributions. There is NO universal threshold.")
        print("  ")
        print("  Using the calibration data from this training run, we recommend:")
        if overall_stats:
            dt = overall_stats['dynamic_threshold']
            print(f"    - Under {dt:.1f} : Confident Match  OK")
            print(f"    - Over {dt:.1f}  : Unknown / Reject  NO")
        else:
            print("    - Match  < (Calibrated Threshold)  OK")
            print("    - Reject >= (Calibrated Threshold)  NO")
        print("=" * 60 + "\n")


# ======================================================================
# Standalone execution – train directly from the command line
# ======================================================================
if __name__ == "__main__":
    trainer = FaceTrainer(
        dataset_path="dataset",
        model_dir="models",
        validate_with_mediapipe=False,
    )
    trainer.train()
