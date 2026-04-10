from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from flask import session
import sqlite3
from datetime import datetime
from flask import make_response, url_for
try:
    from weasyprint import HTML
except Exception:
    HTML = None
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session


# load databasedataset===================================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")


# load model===========================================
try:
    svc = pickle.load(open('svc.pkl','rb'))
except Exception:
    class _DummyModel:
        def predict(self, X):
            return [10]  # Default to 'Common Cold' index as a safe fallback
    svc = _DummyModel()


#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    # Build list of lists robustly without relying on .values/.to_numpy for typing
    pre_list = [list(row) for row in pre_df.itertuples(index=False, name=None)]

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if len(med) > 0 else []

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if len(die) > 0 else []

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist() if len(wrkout) > 0 else []

    return desc, pre_list, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Database setup
conn = sqlite3.connect('healthcare.db', check_same_thread=False)
c = conn.cursor()

# Ensure users table exists
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password_hash TEXT
)''')

# Ensure predictions table exists
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    symptoms TEXT,
    predicted_disease TEXT
)''')
conn.commit()

# Migrate predictions table to include user_id if missing
def ensure_predictions_user_id_column():
    c.execute("PRAGMA table_info(predictions)")
    cols = [row[1] for row in c.fetchall()]
    if 'user_id' not in cols:
        c.execute('ALTER TABLE predictions ADD COLUMN user_id INTEGER')
        conn.commit()

ensure_predictions_user_id_column()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    c.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
    return c.fetchone()

def create_user(username, password):
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def save_prediction(symptoms, predicted_disease):
    user_id = session.get('user_id')
    c.execute('INSERT INTO predictions (date, symptoms, predicted_disease, user_id) VALUES (?, ?, ?, ?)',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), symptoms, predicted_disease, user_id))
    conn.commit()


# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, pre_list, medications_list, rec_diet, workout_list = helper(predicted_disease)
            # Flatten first row of precautions if present
            my_precautions = pre_list[0] if (isinstance(pre_list, list) and len(pre_list) > 0) else []
            # Save prediction to DB
            save_prediction(', '.join(user_symptoms), predicted_disease)
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet,
                                   workout=workout_list)

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    stats = {
        'total_predictions': 1234,
        'common_symptoms': ['fever', 'cough', 'fatigue'],
        'common_diseases': ['Flu', 'COVID-19', 'Diabetes'],
        'disease_counts': {'Flu': 500, 'COVID-19': 400, 'Diabetes': 334},
    }
    return render_template('dashboard.html', **stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = get_user(username) if username else None
        if user and user[2] == hash_password(password or ''):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return render_template('index.html', message='Logged in successfully.')
        return render_template('login.html', message='Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return render_template('index.html', message='Logged out.')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('register.html', message='Username and password are required.')
        ok = create_user(username, password)
        if ok:
            user = get_user(username)
            session['user_id'] = user[0]
            session['username'] = user[1]
            return render_template('index.html', message='Registration successful.')
        return render_template('register.html', message='Username already exists.')
    return render_template('register.html')

@app.route('/medibot', methods=['GET', 'POST'])
def medibot():
    if 'chat_history' not in session:
        session['chat_history'] = []
    chat_history = session['chat_history']
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            chat_history.append({'sender': 'user', 'text': user_input})
            # Simple canned response for demo
            if 'fever' in user_input.lower():
                bot_reply = "If you have a fever, consider resting and staying hydrated. If it persists, consult a doctor."
            else:
                bot_reply = "I'm MediBot! I can help with basic health questions. (Demo response)"
            chat_history.append({'sender': 'bot', 'text': bot_reply})
            session['chat_history'] = chat_history
    return render_template('medibot.html', chat_history=chat_history)

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/image-check', methods=['GET', 'POST'])
def image_check():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # For demo, just echo the filename as a mock result
            result = f"Image '{file.filename}' analyzed. (Demo: No real ML yet)"
    return render_template('image_check.html', result=result)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        return render_template('login.html', message='Please log in to view your history.')
    c.execute('SELECT id, date, symptoms, predicted_disease FROM predictions WHERE user_id = ? ORDER BY date DESC', (user_id,))
    predictions = c.fetchall()
    return render_template('history.html', predictions=predictions)

@app.route('/report/<int:prediction_id>')
def report(prediction_id):
    c.execute('SELECT id, date, symptoms, predicted_disease FROM predictions WHERE id = ?', (prediction_id,))
    row = c.fetchone()
    if not row:
        return 'Prediction not found', 404
    id, date, symptoms, predicted_disease = row
    if HTML is None:
        return render_template('report_pdf.html', date=date, symptoms=symptoms, predicted_disease=predicted_disease)
    # Render HTML for PDF
    html = render_template('report_pdf.html', date=date, symptoms=symptoms, predicted_disease=predicted_disease)
    pdf = HTML(string=html, base_url=url_for('history', _external=True)).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=prediction_report_{id}.pdf'
    return response


if __name__ == '__main__':

    app.run(debug=True)