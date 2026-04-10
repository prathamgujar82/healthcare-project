from flask import Flask, request, render_template, jsonify, session, make_response, url_for, redirect, flash
import pandas as pd
import pickle
import numpy as np
import sqlite3
import os
import hashlib
from datetime import datetime
from constants import symptoms_dict, diseases_list

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey_change_me_in_prod')

# --- Configuration & Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'healthcare.db')
MODEL_PATH = os.path.join(BASE_DIR, 'svc.pkl')

# Load Datasets
try:
    sym_des = pd.read_csv("symtoms_df.csv")
    precautions = pd.read_csv("precautions_df.csv")
    workout = pd.read_csv("workout_df.csv")
    description = pd.read_csv("description.csv")
    medications = pd.read_csv('medications.csv')
    diets = pd.read_csv("diets.csv")
except Exception as e:
    print(f"Error loading CSV files: {e}")

# Load Model
svc = None
try:
    with open(MODEL_PATH, 'rb') as f:
        svc = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Optional: WeasyPrint
try:
    from weasyprint import HTML
except ImportError:
    HTML = None

# --- Database Helpers ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )''')
    
    # Predictions Table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        symptoms TEXT,
        predicted_disease TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    conn.commit()
    conn.close()

init_db()

# --- Logic Helpers ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre_list = [list(row) for row in pre_df.itertuples(index=False, name=None)]

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if len(med) > 0 else []

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if len(die) > 0 else []

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist() if len(wrkout) > 0 else []

    return desc, pre_list, med, die, wrkout

def get_predicted_value(patient_symptoms):
    if not svc:
        return "Model not loaded"
        
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    
    prediction_index = svc.predict([input_vector])[0]
    return diseases_list.get(prediction_index, "Unknown Disease")

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html", symptoms_list=list(symptoms_dict.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms == "Symptoms":
            flash("Please select or enter symptoms.", "warning")
            return redirect(url_for('index'))
            
        # Parse symptoms
        # Support both comma separated string (from text input) and list (from checkboxes if we change form)
        # The original code handled comma separated string.
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [s.strip("[]' ") for s in user_symptoms] # Clean up artifacts
        
        predicted_disease = get_predicted_value(user_symptoms)
        
        dis_des, pre_list, medications_list, rec_diet, workout_list = helper(predicted_disease)
        my_precautions = pre_list[0] if (isinstance(pre_list, list) and len(pre_list) > 0) else []

        # Save to DB if logged in
        if 'user_id' in session:
            conn = get_db_connection()
            conn.execute('INSERT INTO predictions (date, symptoms, predicted_disease, user_id) VALUES (?, ?, ?, ?)',
                         (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ', '.join(user_symptoms), predicted_disease, session['user_id']))
            conn.commit()
            conn.close()

        return render_template('index.html', 
                               predicted_disease=predicted_disease, 
                               dis_des=dis_des,
                               my_precautions=my_precautions, 
                               medications=medications_list, 
                               my_diet=rec_diet,
                               workout=workout_list,
                               scroll_to_result=True,
                               symptoms_list=list(symptoms_dict.keys()))

@app.route('/dashboard')
def dashboard():
    # Mock stats
    stats = {
        'total_predictions': 1234,
        'common_symptoms': ['fever', 'cough', 'fatigue'],
        'common_diseases': ['Flu', 'COVID-19', 'Diabetes'],
        'disease_counts': {'Flu': 500, 'COVID-19': 400, 'Diabetes': 334},
    }
    return render_template('dashboard.html', **stats)

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and user['password_hash'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required.', 'warning')
            return render_template('register.html')
            
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                         (username, hash_password(password)))
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('index'))

# Other Routes
@app.route('/medibot', methods=['GET', 'POST'])
def medibot():
    if 'chat_history' not in session:
        session['chat_history'] = []
    chat_history = session['chat_history']
    
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            chat_history.append({'sender': 'user', 'text': user_input})
            # Simple logic
            if 'fever' in user_input.lower():
                reply = "Rest and hydration are key for fever. Consult a doctor if it's high."
            else:
                reply = "I am MedAI Bot. I can help with general health queries."
            chat_history.append({'sender': 'bot', 'text': reply})
            session['chat_history'] = chat_history
            
    return render_template('medibot.html', chat_history=chat_history)

@app.route('/image-check', methods=['GET', 'POST'])
def image_check():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Mock
            result = f"Image '{file.filename}' received. Analysis: Healthy (Demo)."
    return render_template('image_check.html', result=result)

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in to view history.", "warning")
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    predictions = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY date DESC', 
                               (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', predictions=predictions)

@app.route('/report/<int:prediction_id>')
def download_report(prediction_id):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    conn.close()
    
    if not row:
        return 'Prediction not found', 404
        
    if HTML:
        html = render_template('report_pdf.html', 
                               date=row['date'], 
                               symptoms=row['symptoms'], 
                               predicted_disease=row['predicted_disease'])
        pdf = HTML(string=html).write_pdf()
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{prediction_id}.pdf'
        return response
    else:
        return "PDF generation library (WeasyPrint) not available."

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/contact')
def contact(): return render_template('contact.html')

@app.route('/developer')
def developer(): return render_template('developer.html')

@app.route('/blog')
def blog(): return render_template('blog.html')

if __name__ == '__main__':
    app.run(debug=True)
