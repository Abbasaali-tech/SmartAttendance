import threading
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.face_data_collector import FaceDataCollector
from models.face_trainer import FaceTrainer
from models.face_recognizer import FaceRecognizer
from database.attendance_db import AttendanceDB


class AttendanceApp:
    """
    Tkinter GUI for the Smart Attendance System.
    Provides a simple dashboard and connects to backend modules.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("900x520")
        self.root.minsize(820, 480)

        self.db = AttendanceDB()
        self.recognizer = None
        self.recognition_thread = None
        self.recognition_stop_event = None

        self.status_var = tk.StringVar(value="Ready")
        self._build_layout()

    def _build_layout(self):
        self.root.configure(bg="#f4f6f8")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        header = tk.Frame(self.root, bg="#1f2937", height=72)
        header.pack(fill="x")
        header_label = tk.Label(
            header,
            text="Smart Attendance Dashboard",
            bg="#1f2937",
            fg="#f9fafb",
            font=("Segoe UI", 18, "bold"),
            padx=20,
            pady=18,
        )
        header_label.pack(anchor="w")

        content = tk.Frame(self.root, bg="#f4f6f8")
        content.pack(fill="both", expand=True, padx=20, pady=18)

        left_panel = tk.Frame(content, bg="#f4f6f8")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right_panel = tk.Frame(content, bg="#f4f6f8")
        right_panel.pack(side="right", fill="y")

        self._build_section(
            left_panel,
            "Student Registration",
            "Capture student faces and build dataset.",
            "Register Student",
            self._run_register,
        )

        self._build_section(
            left_panel,
            "Training",
            "Train LBPH model and update confidence stats.",
            "Train Model",
            self._run_training,
        )

        self._build_section(
            left_panel,
            "Attendance",
            "Start real-time face recognition and mark attendance.",
            "Start Attendance",
            self._run_recognition,
            button2_label="Stop Attendance",
            button2_command=self._stop_recognition,
        )

        self._build_section(
            right_panel,
            "Records",
            "View and export attendance records.",
            "View Attendance",
            self._run_view_attendance,
            button2_label="Export CSV",
            button2_command=self._run_export_csv,
        )

        self._build_log_panel(right_panel)
        self._start_log_refresh()

        status_bar = tk.Frame(self.root, bg="#e5e7eb", height=30)
        status_bar.pack(fill="x", side="bottom")
        status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg="#e5e7eb",
            fg="#374151",
            anchor="w",
            padx=12,
        )
        status_label.pack(fill="x")

    def _build_section(
        self,
        parent,
        title,
        subtitle,
        button1_label,
        button1_command,
        button2_label=None,
        button2_command=None,
    ):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief="solid")
        card.pack(fill="x", pady=10)

        title_label = tk.Label(
            card,
            text=title,
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 13, "bold"),
            padx=14,
            pady=10,
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            card,
            text=subtitle,
            bg="#ffffff",
            fg="#6b7280",
            font=("Segoe UI", 10),
            padx=14,
        )
        subtitle_label.pack(anchor="w")

        button_frame = tk.Frame(card, bg="#ffffff")
        button_frame.pack(fill="x", padx=14, pady=12)

        btn_primary = ttk.Button(button_frame, text=button1_label, command=button1_command)
        btn_primary.pack(side="left")

        if button2_label and button2_command:
            btn_secondary = ttk.Button(button_frame, text=button2_label, command=button2_command)
            btn_secondary.pack(side="left", padx=8)

    def _build_log_panel(self, parent):
        card = tk.Frame(parent, bg="#ffffff", bd=1, relief="solid")
        card.pack(fill="both", expand=True, pady=10)

        title_label = tk.Label(
            card,
            text="Live Activity Log",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 12, "bold"),
            padx=14,
            pady=10,
        )
        title_label.pack(anchor="w")

        form_frame = tk.Frame(card, bg="#ffffff")
        form_frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(form_frame, text="Student Name:", bg="#ffffff", fg="#374151").grid(row=0, column=0, sticky="w")
        self.student_name_entry = ttk.Entry(form_frame, width=22)
        self.student_name_entry.grid(row=0, column=1, padx=(8, 16), pady=4, sticky="w")

        tk.Label(form_frame, text="Student ID:", bg="#ffffff", fg="#374151").grid(row=0, column=2, sticky="w")
        self.student_id_entry = ttk.Entry(form_frame, width=16)
        self.student_id_entry.grid(row=0, column=3, padx=(8, 0), pady=4, sticky="w")

        log_frame = tk.Frame(card, bg="#ffffff")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            bg="#f9fafb",
            fg="#111827",
            font=("Segoe UI", 9),
            relief="flat",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _set_status(self, message: str):
        self.status_var.set(message)
        self.root.update_idletasks()

    def _append_log(self, message: str):
        def write():
            self.log_text.insert("end", f"{message}\n")
            self.log_text.see("end")
        self.root.after(0, write)

    def _start_log_refresh(self):
        def refresh():
            if self.log_text.winfo_exists():
                self.log_text.see("end")
                self.root.after(1000, refresh)
        refresh()

    def _run_in_thread(self, target, status_message):
        def runner():
            self._set_status(status_message)
            self._append_log(status_message)
            try:
                target()
                self._set_status("Ready")
                self._append_log("Ready")
            except Exception as exc:
                self._set_status("Error occurred")
                self._append_log(f"Error: {exc}")
                error_msg = str(exc)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return thread

    def _run_register(self):
        name = self.student_name_entry.get().strip()
        student_id = self.student_id_entry.get().strip()
        if not name or not student_id:
            messagebox.showwarning("Register Student", "Please enter both Student Name and Student ID.")
            return

        def task():
            collector = FaceDataCollector(
                dataset_path="dataset",
                max_samples=50,
                min_detection_confidence=0.6,
                student_name=name,
                student_id=student_id,
            )
            collector.collect_faces()

        self._run_in_thread(task, "Registering student...")

    def _run_training(self):
        def task():
            trainer = FaceTrainer(
                dataset_path="dataset",
                model_dir="models",
                validate_with_mediapipe=True,
            )
            trainer.train()

        self._run_in_thread(task, "Training model...")

    def _run_recognition(self):
        def task():
            self.recognizer = FaceRecognizer(logger_callback=self._append_log)
            self.recognizer.recognize_faces(self.db, self.recognition_stop_event)

        if self.recognition_thread and self.recognition_thread.is_alive():
            messagebox.showinfo("Attendance", "Attendance is already running.")
            return

        self.recognition_stop_event = threading.Event()
        self.recognition_thread = self._run_in_thread(task, "Starting attendance...")

    def _stop_recognition(self):
        if not self.recognition_thread or not self.recognition_thread.is_alive():
            messagebox.showinfo("Attendance", "Attendance is not running.")
            return
        self._set_status("Stopping attendance...")
        self._append_log("Stopping attendance...")
        self.recognition_stop_event.set()

    def _run_view_attendance(self):
        def task():
            window = tk.Toplevel(self.root)
            window.title("Attendance Records")
            window.geometry("760x480")

            filter_frame = tk.Frame(window, bg="#f9fafb")
            filter_frame.pack(fill="x", padx=10, pady=10)

            tk.Label(filter_frame, text="Date (YYYY-MM-DD):", bg="#f9fafb").grid(row=0, column=0, padx=6, pady=4, sticky="w")
            date_entry = ttk.Entry(filter_frame, width=16)
            date_entry.grid(row=0, column=1, padx=6, pady=4, sticky="w")

            tk.Label(filter_frame, text="Student (ID or Name):", bg="#f9fafb").grid(row=0, column=2, padx=6, pady=4, sticky="w")
            student_entry = ttk.Entry(filter_frame, width=22)
            student_entry.grid(row=0, column=3, padx=6, pady=4, sticky="w")

            def load_records():
                date_filter = date_entry.get().strip()
                student_filter = student_entry.get().strip().lower()
                records = self.db.fetch_attendance(date_filter if date_filter else None)

                tree.delete(*tree.get_children())
                for row in records:
                    student_id, student_name, date_value, time_value = row
                    if student_filter and student_filter not in student_id.lower() and student_filter not in student_name.lower():
                        continue
                    tree.insert("", "end", values=row)

                if not tree.get_children():
                    self._append_log("No attendance records matched the filter.")

            def schedule_refresh():
                if window.winfo_exists():
                    load_records()
                    window.after(5000, schedule_refresh)

            ttk.Button(filter_frame, text="Apply Filters", command=load_records).grid(row=0, column=4, padx=6, pady=4)

            tree = ttk.Treeview(window, columns=("id", "name", "date", "time"), show="headings")
            tree.heading("id", text="Student ID")
            tree.heading("name", text="Student Name")
            tree.heading("date", text="Date")
            tree.heading("time", text="Time")
            tree.pack(fill="both", expand=True, padx=10, pady=10)

            load_records()
            schedule_refresh()

        self._run_in_thread(task, "Loading attendance records...")

    def _run_export_csv(self):
        def task():
            filename = filedialog.asksaveasfilename(
                title="Export Attendance",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if not filename:
                return
            if not self.db.export_to_csv(filename):
                raise RuntimeError("CSV export failed or no records available.")

        self._run_in_thread(task, "Exporting attendance...")


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
