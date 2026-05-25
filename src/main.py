"""
Smart Attendance System - Main Driver
=====================================================================
Author:  Abbas
Description:
    Interactive entry point for the Smart Attendance System.
    Presents a menu so the user can choose between:
        1. Face Registration  (capture face images)
        2. Model Training     (train the LBPH recognizer)
        3. Exit
"""

import sys
from data.face_data_collector import FaceDataCollector
from models.face_trainer import FaceTrainer
from models.face_recognizer import FaceRecognizer

def show_menu() -> str:
    """Prints the main menu and returns the user's choice."""
    print("\n" + "=" * 50)
    print("      SMART ATTENDANCE SYSTEM – MAIN MENU")
    print("=" * 50)
    print("  [1]  Register a new student (face capture)")
    print("  [2]  Train face recognition model (LBPH)")
    print("  [3]  Run Face Recognition & Mark Attendance")
    print("  [4]  Launch GUI Dashboard")
    print("  [5]  Run Health Check")
    print("  [q]  Quit")
    print("=" * 50)
    return input("  Select an option: ").strip().lower()


def register_student():
    """Launches the face registration pipeline."""
    collector = FaceDataCollector(
        dataset_path="dataset",
        max_samples=50,
        min_detection_confidence=0.6,
    )
    collector.collect_faces()


def train_model():
    """Launches the LBPH training pipeline."""
    trainer = FaceTrainer(
        dataset_path="dataset",
        model_dir="models",
        validate_with_mediapipe=True,
    )
    trainer.train()

def run_recognition_and_attendance():
    """Launches the real-time face recognition and attendance marking pipeline."""
    from database.attendance_db import AttendanceDB
    recognizer = FaceRecognizer()
    attendance_db = AttendanceDB()
    recognizer.recognize_faces(attendance_db)


def launch_gui():
    """Launches the Tkinter GUI dashboard."""
    from gui.attendance_app import AttendanceApp
    import tkinter as tk
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


def run_health_check():
    """Verify model files, database tables, and camera access."""
    import os
    import cv2
    from database.attendance_db import AttendanceDB

    print("\n" + "=" * 50)
    print("         SMART ATTENDANCE HEALTH CHECK")
    print("=" * 50)

    model_path = os.path.join("models", "lbph_model.yml")
    labels_path = os.path.join("models", "labels.pkl")
    stats_path = os.path.join("models", "confidence_stats.pkl")

    def report(name, ok, detail=""):
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name} {detail}")

    report("Model file", os.path.exists(model_path), f"({model_path})")
    report("Labels file", os.path.exists(labels_path), f"({labels_path})")
    report("Confidence stats", os.path.exists(stats_path), f"({stats_path})")

    try:
        db = AttendanceDB()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        required = {"students", "attendance", "unknown_faces"}
        missing = required - tables
        report("Database tables", not missing, f"(missing: {', '.join(sorted(missing))})" if missing else "")
    except Exception as exc:
        report("Database tables", False, f"({exc})")

    cap = cv2.VideoCapture(0)
    camera_ok = cap.isOpened()
    if camera_ok:
        cap.release()
    report("Camera access", camera_ok, "(index 0)")

    print("=" * 50 + "\n")

def main():
    try:
        while True:
            choice = show_menu()

            if choice == "1":
                register_student()
            elif choice == "2":
                train_model()
            elif choice == "3":
                run_recognition_and_attendance()
            elif choice == "4":
                launch_gui()
            elif choice == "5":
                run_health_check()
            elif choice == "q":
                print("\n[INFO] Goodbye!")
                break
            else:
                print("[WARNING] Invalid option. Please enter 1, 2, 3, 4, 5, or q.")

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] System encountered an unexpected issue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
