# HealthsenseAI | Healthcare Assessment & Wellness System

This guide explains how to set up and run the **HealthsenseAI Professional Clinical Assessment System** on a new computer.

## 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system.
- Check your version: `python --version`

## 2. Installation Steps

### Step 1: Open Terminal in VS Code
Open the project folder in VS Code. Open a new terminal (`Ctrl + ` ` or `Terminal > New Terminal`).

### Step 2: Create a Virtual Environment
This keeps the project dependencies isolated.
```powershell
# Create the environment
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If you see "Execution Policy" error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` then try again.)*

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

## 3. Running the Application
Once the installation is complete, you can start the application:

```powershell
python main.py
```

### Accessing the Web Interface
1. Once the terminal says `Serving Flask app 'main'`, open your browser.
2. Go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔬 Project Architecture
* **`main.py`**: The primary Flask application entry point with authentication, appointments booking, and calorie tracker routes.
* **`database.py`**: SQLite database controller handling profiles, diagnosis history logs, and calorie summaries.
* **`requirements.txt`**: Contains all necessary libraries (Flask, Scikit-learn, Pandas, etc.).
* **`data/`**: Directory containing database schemas (`database.db`), Trained SVM Classifier model (`svc.pkl`), and diagnostic details datasets (descriptions, precautions, medications, workouts).
