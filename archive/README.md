# Healthcare Assistant

A Flask app that predicts diseases from symptoms and provides basic recommendations, login/register, history, and PDF reports (optional).

## Setup (Windows PowerShell)

```powershell
cd C:\Users\harsh\OneDrive\Documents\healthcare
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:FLASK_APP="main.py"
python .\main.py
```

Open http://127.0.0.1:5000/

If WeasyPrint isn't installed or fails, the PDF endpoint will return an HTML page instead of a PDF.

## Project Structure

- `main.py`: Flask app
- `templates/`: Jinja2 templates
- `static/css/style.css`: Minimal styles
- CSVs and `svc.pkl`: Data and model (optional; app has fallback)

## Notes

- SQLite DB `healthcare.db` is created automatically.
- Login/register creates users in the local DB; not for production use.