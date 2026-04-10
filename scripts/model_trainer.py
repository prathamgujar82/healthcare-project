import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Loading dataset...")
    try:
        df = pd.read_csv("Training.csv")
    except FileNotFoundError:
        logging.error("Training.csv not found!")
        return

    # Prepare data
    # The last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Map prognosis strings to integers if they are not already (based on main.py dict)
    # Actually main.py used a dictionary `diseases_list` which keys were integers.
    # But `Training.csv` has string labels likely.
    # usage in main.py: `diseases_list[svc.predict([input_vector])[0]]`
    # This implies the model output is an INDEX (int).
    # So we need to map the string labels in y to the integers in diseases_list.
    
    # Let's import the diseases_list from constants to reverse map it.
    try:
        from constants import diseases_list
    except ImportError:
        logging.error("constants.py not found or diseases_list missing.")
        return

    # Create reverse mapping: Name -> ID
    disease_to_id = {v: k for k, v in diseases_list.items()}
    
    # Map y
    try:
        y = y.map(disease_to_id)
    except Exception as e:
        logging.error(f"Error mapping labels: {e}")
        return

    # Check for NaN in y (mismatched disease names)
    if y.isnull().any():
        logging.warning("Some disease names in CSV did not match constants.diseases_list. Dropping them.")
        # Filter out rows where y is NaN
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
    
    # Train test split for verification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info("Training SVC model...")
    svc = SVC()
    svc.fit(X_train, y_train)
    
    # Evaluate
    predictions = svc.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    logging.info(f"Model accuracy: {acc:.4f}")
    
    # Retrain on full data for production
    logging.info("Retraining on full dataset...")
    svc.fit(X, y)
    
    # Save model
    logging.info("Saving model to svc.pkl...")
    with open('svc.pkl', 'wb') as f:
        pickle.dump(svc, f)
    
    logging.info("Done.")

if __name__ == "__main__":
    train_model()
