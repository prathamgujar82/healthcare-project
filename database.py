import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date

DB_PATH = 'data/database.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Symptom diagnostic history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_disease TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            probability REAL NOT NULL,
            description TEXT,
            precautions TEXT,
            medications TEXT,
            diets TEXT,
            workout TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            doctor_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'Scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Diet logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diet_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            food_name TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            calories INTEGER NOT NULL,
            protein REAL DEFAULT 0,
            carbs REAL DEFAULT 0,
            fat REAL DEFAULT 0,
            logged_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Exercise logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity TEXT NOT NULL,
            duration INTEGER NOT NULL,
            calories_burned INTEGER NOT NULL,
            logged_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# User operations
def register_user(name, email, password):
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (name, email, hashed_password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(email, password):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        return dict(user)
    return None

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

# Diagnosis operations
def save_diagnosis(user_id, predicted_disease, symptoms, probability, description, precautions, medications, diets, workout):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diagnostic_history 
        (user_id, predicted_disease, symptoms, probability, description, precautions, medications, diets, workout) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, 
        predicted_disease, 
        symptoms, 
        probability, 
        description, 
        precautions, 
        medications, 
        diets, 
        workout
    ))
    conn.commit()
    conn.close()

def get_user_diagnoses(user_id):
    conn = get_db_connection()
    diagnoses = conn.execute(
        'SELECT * FROM diagnostic_history WHERE user_id = ? ORDER BY created_at DESC', 
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(d) for d in diagnoses]

def get_diagnosis_by_id(diag_id):
    conn = get_db_connection()
    diagnosis = conn.execute(
        'SELECT * FROM diagnostic_history WHERE id = ?', 
        (diag_id,)
    ).fetchone()
    conn.close()
    if diagnosis:
        return dict(diagnosis)
    return None

# Doctor Appointment booking operations
def book_appointment(user_id, doctor_name, specialty, appt_date, appt_time, reason):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO appointments (user_id, doctor_name, specialty, appointment_date, appointment_time, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, doctor_name, specialty, appt_date, appt_time, reason))
    conn.commit()
    conn.close()

def get_user_appointments(user_id):
    conn = get_db_connection()
    appointments = conn.execute(
        'SELECT * FROM appointments WHERE user_id = ? ORDER BY appointment_date ASC, appointment_time ASC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(a) for a in appointments]

def cancel_appointment(appt_id, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE appointments SET status = 'Cancelled' WHERE id = ? AND user_id = ?",
        (appt_id, user_id)
    )
    conn.commit()
    conn.close()

# Diet logs operations
def log_food(user_id, food_name, meal_type, calories, protein, carbs, fat, logged_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diet_logs (user_id, food_name, meal_type, calories, protein, carbs, fat, logged_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, food_name, meal_type, int(calories), float(protein), float(carbs), float(fat), logged_date))
    conn.commit()
    conn.close()

def get_food_logs(user_id, logged_date):
    conn = get_db_connection()
    logs = conn.execute(
        'SELECT * FROM diet_logs WHERE user_id = ? AND logged_date = ? ORDER BY created_at ASC',
        (user_id, logged_date)
    ).fetchall()
    conn.close()
    return [dict(l) for l in logs]

def delete_food_log(log_id, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM diet_logs WHERE id = ? AND user_id = ?', (log_id, user_id))
    conn.commit()
    conn.close()

# Exercise logs operations
def log_exercise(user_id, activity, duration, calories_burned, logged_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO exercise_logs (user_id, activity, duration, calories_burned, logged_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, activity, int(duration), int(calories_burned), logged_date))
    conn.commit()
    conn.close()

def get_exercise_logs(user_id, logged_date):
    conn = get_db_connection()
    logs = conn.execute(
        'SELECT * FROM exercise_logs WHERE user_id = ? AND logged_date = ? ORDER BY created_at ASC',
        (user_id, logged_date)
    ).fetchall()
    conn.close()
    return [dict(l) for l in logs]

def delete_exercise_log(log_id, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM exercise_logs WHERE id = ? AND user_id = ?', (log_id, user_id))
    conn.commit()
    conn.close()

# Stats and Dashboard metrics
def get_dashboard_stats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = date.today().isoformat()
    
    # Total diagnoses
    total_diag = cursor.execute('SELECT COUNT(*) FROM diagnostic_history WHERE user_id = ?', (user_id,)).fetchone()[0]
    
    # Upcoming appointments count
    upcoming_appt = cursor.execute(
        "SELECT COUNT(*) FROM appointments WHERE user_id = ? AND status = 'Scheduled' AND appointment_date >= ?",
        (user_id, today_str)
    ).fetchone()[0]
    
    # Diet calories consumed today
    calories_consumed = cursor.execute(
        "SELECT SUM(calories) FROM diet_logs WHERE user_id = ? AND logged_date = ?",
        (user_id, today_str)
    ).fetchone()[0]
    calories_consumed = calories_consumed if calories_consumed else 0

    # Exercise calories burned today
    calories_burned = cursor.execute(
        "SELECT SUM(calories_burned) FROM exercise_logs WHERE user_id = ? AND logged_date = ?",
        (user_id, today_str)
    ).fetchone()[0]
    calories_burned = calories_burned if calories_burned else 0
    
    conn.close()
    return {
        "total_diagnoses": total_diag,
        "upcoming_appointments": upcoming_appt,
        "today_consumed": calories_consumed,
        "today_burned": calories_burned,
        "today_net": calories_consumed - calories_burned
    }
