import sqlite3
import os
import csv
from datetime import datetime

class AttendanceDB:
    """
    Handles SQLite database operations for the Smart Attendance System.
    Includes managing students, marking attendance, fetching records,
    and exporting to CSV.
    """

    def __init__(self, db_path="database/attendance.db"):
        self.db_path = db_path
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _get_connection(self):
        """Helper method to get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_db(self):
        """Creates the necessary tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create students table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT UNIQUE NOT NULL,
                        student_name TEXT NOT NULL
                    )
                ''')
                
                # Create attendance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        student_name TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        UNIQUE(student_id, date),
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS unknown_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_path TEXT NOT NULL,
                        detected_at TEXT NOT NULL
                    )
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_attendance_student
                    ON attendance(student_id)
                ''')

                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance(date)
                ''')
                
                conn.commit()
        except sqlite3.Error as e:
            print(f"[ERROR] Database initialization failed: {e}")

    def add_student(self, student_id: str, student_name: str) -> bool:
        """
        Adds a new student to the database.
        Returns True if successful, False if the student already exists.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO students (student_id, student_name)
                    VALUES (?, ?)
                ''', (student_id, student_name))

                cursor.execute('''
                    UPDATE students SET student_name = ? WHERE student_id = ?
                ''', (student_name, student_id))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to add student: {e}")
            return False

    def mark_attendance(self, student_id: str, student_name: str) -> bool:
        """
        Marks attendance for a student. Prevents duplicate marking per day.
        Returns True if marked successfully, False if already marked or error.
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check for duplicate attendance on the same day
                cursor.execute('''
                    SELECT id FROM attendance 
                    WHERE student_id = ? AND date = ?
                ''', (student_id, current_date))
                
                if cursor.fetchone() is not None:
                    # Already marked today
                    return False
                    
                # Insert the attendance record
                cursor.execute('''
                    INSERT INTO attendance (student_id, student_name, date, time)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, student_name, current_date, current_time))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to mark attendance: {e}")
            return False

    def fetch_attendance(self, date: str = None) -> list:
        """
        Fetches attendance records. If a date is provided (YYYY-MM-DD),
        fetches only for that date. Otherwise, fetches all records.
        """
        records = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if date:
                    cursor.execute('''
                        SELECT attendance.student_id, students.student_name, attendance.date, attendance.time
                        FROM attendance
                        JOIN students ON attendance.student_id = students.student_id
                        WHERE attendance.date = ?
                        ORDER BY time ASC
                    ''', (date,))
                else:
                    cursor.execute('''
                        SELECT attendance.student_id, students.student_name, attendance.date, attendance.time
                        FROM attendance
                        JOIN students ON attendance.student_id = students.student_id
                        ORDER BY date DESC, time ASC
                    ''')
                records = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to fetch attendance: {e}")
            
        return records

    def export_to_csv(self, filename: str = "attendance_export.csv", date: str = None) -> bool:
        """
        Exports the attendance records to a CSV file.
        Can optionally filter by date (YYYY-MM-DD).
        """
        records = self.fetch_attendance(date)
        if not records:
            print("[WARNING] No records found to export.")
            return False

        try:
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Student ID", "Student Name", "Date", "Time"])
                writer.writerows(records)
            print(f"[INFO] Successfully exported attendance to {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] CSV Export failed: {e}")
            return False

    def log_unknown(self, snapshot_path: str) -> bool:
        """
        Logs an unknown face detection with a timestamp and snapshot path.
        """
        detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO unknown_faces (snapshot_path, detected_at)
                    VALUES (?, ?)
                ''', (snapshot_path, detected_at))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to log unknown face: {e}")
            return False
