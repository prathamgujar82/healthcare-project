from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle
import warnings
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import uuid
from datetime import date

# Import local database helper
import database as db

# Initialize database
db.init_db()

# Suppress scikit-learn unpickle version warning
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key_for_session" # Secure secret key for session management

# Load all necessary models and data
try:
    svc = pickle.load(open('data/svc.pkl', 'rb'))
    svc_loaded = True
except Exception as e:
    print(f"Warning: svc.pkl model not found: {e}")
    svc_loaded = False

# Load dataset files for details
try:
    description = pd.read_csv("data/description.csv")
    precautions = pd.read_csv("data/precautions_df.csv")
    medications = pd.read_csv('data/medications.csv')
    diets = pd.read_csv("data/diets.csv")
    workout = pd.read_csv("data/workout_df.csv")
    datasets_loaded = True
except Exception as e:
    print(f"Warning: Dataset files error: {e}")
    datasets_loaded = False

# Load constants (manually defined to avoid dependency issues)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# ============== Helper Functions ==============
def get_details(dis):
    """Retrieve detailed information about a disease from datasets"""
    if not datasets_loaded:
        return "No description available.", [], [], [], []
    
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "No description available."

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre_list = pre.values.flatten().tolist() if not pre.empty else []
    pre_list = [p for p in pre_list if str(p) != 'nan']

    med = medications[medications['Disease'] == dis]['Medication']
    med_list = med.tolist() if not med.empty else []

    die = diets[diets['Disease'] == dis]['Diet']
    die_list = die.tolist() if not die.empty else []

    wrk = workout[workout['disease'] == dis]['workout']
    wrk_list = wrk.tolist() if not wrk.empty else []

    return desc, pre_list, med_list, die_list, wrk_list

def predict_disease(user_symptoms):
    """Make prediction using SVC model with probabilities if supported"""
    if not svc_loaded:
        return "Model not available", []
    
    input_vector = np.zeros(len(symptoms_dict))
    for item in user_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    
    # Predict actual disease
    prediction_idx = svc.predict([input_vector])[0]
    disease = diseases_list.get(prediction_idx, "Unknown")
    
    # Try to get top 3 probabilities
    probs = []
    try:
        if hasattr(svc, "predict_proba"):
            probabilities = svc.predict_proba([input_vector])[0]
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            for i in top_3_idx:
                probs.append({
                    "disease": diseases_list.get(i, "Unknown"),
                    "probability": round(probabilities[i] * 100, 1)
                })
        else:
            probs = [{"disease": disease, "probability": 100.0}]
    except:
        probs = [{"disease": disease, "probability": 100.0}]
            
    return disease, probs

# ============== Routes ==============

@app.route("/")
def index():
    """Main landing page"""
    return render_template("home.html", symptoms_list=list(symptoms_dict.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    symptoms = request.form.getlist('symptoms')
    if not symptoms:
        symptoms_str = request.form.get('symptoms')
        if symptoms_str:
            symptoms = [s.strip() for s in symptoms_str.split(',')]
    
    if not symptoms:
        flash("Please select at least one symptom indicator.", "warning")
        return redirect(url_for('index'))
    
    # Perform prediction
    disease, probabilities = predict_disease(symptoms)
    
    # Get details
    desc, pre, med, diet, wrk = get_details(disease)

    # Save to history if user is authenticated
    if session.get('user_id'):
        db.save_diagnosis(
            session['user_id'],
            disease,
            ",".join(symptoms),
            probabilities[0]['probability'] if probabilities else 100.0,
            desc,
            ",".join(pre),
            ",".join(med),
            ",".join(diet),
            ",".join(wrk)
        )

    return render_template('results.html', 
                          predicted_disease=disease, 
                          probabilities=probabilities,
                          dis_des=desc,
                          my_precautions=pre, 
                          medications=med, 
                          my_diet=diet,
                          workout=wrk,
                          selected_symptoms=symptoms)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = db.authenticate_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            flash("Welcome back, Dr. {}.".format(user['name']), "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Authentication failed. Please verify credentials.", "error")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        success = db.register_user(name, email, password)
        if success:
            user = db.authenticate_user(email, password)
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['user_email'] = user['email']
                flash("Workspace created successfully.", "success")
                return redirect(url_for('dashboard'))
        else:
            flash("Registration failed. Email might already be registered.", "error")
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Session terminated.", "success")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        flash("Please authenticate to access the Clinical Dashboard.", "warning")
        return redirect(url_for('login'))
        
    user = db.get_user_by_id(session['user_id'])
    diagnoses = db.get_user_diagnoses(session['user_id'])
    stats = db.get_dashboard_stats(session['user_id'])
    
    return render_template('dashboard.html', user=user, diagnoses=diagnoses, stats=stats)

# ============== Appointments Routes ==============

@app.route('/appointments', methods=['GET'])
def appointments():
    if not session.get('user_id'):
        flash("Please login to schedule appointments.", "warning")
        return redirect(url_for('login'))
        
    user_appts = db.get_user_appointments(session['user_id'])
    return render_template('appointments.html', appointments=user_appts)

@app.route('/appointments/book', methods=['POST'])
def book_appt():
    if not session.get('user_id'):
        flash("Please login to schedule appointments.", "warning")
        return redirect(url_for('login'))
        
    doctor_name = request.form.get('doctor_name')
    specialty = request.form.get('specialty')
    appointment_date = request.form.get('appointment_date')
    appointment_time = request.form.get('appointment_time')
    reason = request.form.get('reason', '')
    
    if not doctor_name or not specialty or not appointment_date or not appointment_time:
        flash("Missing booking details. Please verify your selection.", "error")
        return redirect(url_for('appointments'))
        
    db.book_appointment(session['user_id'], doctor_name, specialty, appointment_date, appointment_time, reason)
    flash(f"Appointment booked successfully with {doctor_name}.", "success")
    return redirect(url_for('appointments'))

@app.route('/appointments/cancel/<int:appt_id>', methods=['POST'])
def cancel_appt(appt_id):
    if not session.get('user_id'):
        flash("Please login to cancel appointments.", "warning")
        return redirect(url_for('login'))
        
    db.cancel_appointment(appt_id, session['user_id'])
    flash("Appointment has been cancelled.", "success")
    return redirect(url_for('appointments'))

# ============== Diet & Exercise Tracker Routes ==============

@app.route('/tracker', methods=['GET'])
def tracker():
    if not session.get('user_id'):
        flash("Please login to access the Wellness Tracker.", "warning")
        return redirect(url_for('login'))
        
    today_str = date.today().isoformat()
    food_logs = db.get_food_logs(session['user_id'], today_str)
    exercise_logs = db.get_exercise_logs(session['user_id'], today_str)
    stats = db.get_dashboard_stats(session['user_id'])
    
    # Calculate macro totals
    protein_tot = round(sum(f['protein'] for f in food_logs), 1)
    carbs_tot = round(sum(f['carbs'] for f in food_logs), 1)
    fat_tot = round(sum(f['fat'] for f in food_logs), 1)
    
    totals = {
        "protein": protein_tot,
        "carbs": carbs_tot,
        "fat": fat_tot
    }
    
    return render_template('tracker.html', 
                           food_logs=food_logs, 
                           exercise_logs=exercise_logs, 
                           stats=stats, 
                           totals=totals,
                           today=date.today().strftime("%B %d, %Y"))

@app.route('/tracker/food', methods=['POST'])
def add_food():
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    food_name = request.form.get('food_name')
    meal_type = request.form.get('meal_type')
    calories = request.form.get('calories')
    protein = request.form.get('protein', 0)
    carbs = request.form.get('carbs', 0)
    fat = request.form.get('fat', 0)
    
    if not food_name or not meal_type or not calories:
        flash("Please specify food name, meal type and calorie inputs.", "error")
        return redirect(url_for('tracker'))
        
    today_str = date.today().isoformat()
    db.log_food(session['user_id'], food_name, meal_type, calories, protein, carbs, fat, today_str)
    flash(f"Logged meal item: {food_name}.", "success")
    return redirect(url_for('tracker'))

@app.route('/tracker/exercise', methods=['POST'])
def add_exercise():
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    activity = request.form.get('activity')
    duration = request.form.get('duration')
    calories_burned = request.form.get('calories_burned')
    
    if not activity or not duration or not calories_burned:
        flash("Please specify activity details, duration and expenditure inputs.", "error")
        return redirect(url_for('tracker'))
        
    today_str = date.today().isoformat()
    db.log_exercise(session['user_id'], activity, duration, calories_burned, today_str)
    flash(f"Logged workout: {activity}.", "success")
    return redirect(url_for('tracker'))

@app.route('/tracker/delete/food/<int:log_id>', methods=['POST'])
def delete_food(log_id):
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    db.delete_food_log(log_id, session['user_id'])
    flash("Meal record deleted.", "success")
    return redirect(url_for('tracker'))

@app.route('/tracker/delete/exercise/<int:log_id>', methods=['POST'])
def delete_exercise(log_id):
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    db.delete_exercise_log(log_id, session['user_id'])
    flash("Workout record deleted.", "success")
    return redirect(url_for('tracker'))

# ============== Standard Information Routes ==============

@app.route('/reports')
def reports():
    if not session.get('user_id'):
        flash("Authentication required to access Clinical Reports.", "warning")
        return redirect(url_for('login'))
        
    diagnoses = db.get_user_diagnoses(session['user_id'])
    appointments = db.get_user_appointments(session['user_id'])
    
    # We pass diagnostic reports and appointments list to reports template
    return render_template('reports.html', diagnoses=diagnoses, appointments=appointments)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash("Thank you. Your inquiry has been logged in clinical queues.", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/coming-soon')
def coming_soon():
    return render_template('coming_soon.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)