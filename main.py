# Make sure to import HTTPException
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import sklearn


# Load trained model
best_model_gb = joblib.load("GBR_model.pkl")

# Initialize FastAPI
app = FastAPI()

# ... (your CORSMiddleware code is fine) ...

# Templates
templates = Jinja2Templates(directory="templates")

# Input model
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- UPDATED ROUTE ---
@app.post("/predict_single")
def predict_single(input_data: InsuranceInput):
    try:
        # --- 1. Define Mappings ---
        # Based on your "working" code snippet
        sex_mapping = {"male": "M", "female": "F"}
        smoker_mapping = {"yes": "yes", "no": "no"}
        
        # --- 2. Get and Validate Inputs ---
        sex_raw = input_data.sex.lower()
        smoker_raw = input_data.smoker.lower()
        region_raw = input_data.region.lower()

        sex_mapped = sex_mapping.get(sex_raw)
        smoker_mapped = smoker_mapping.get(smoker_raw)

        # Validate regions
        valid_regions = ["northeast", "northwest", "southeast", "southwest"]
        if region_raw not in valid_regions:
            # Send a 422 Unprocessable Entity error
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid region. Must be one of {valid_regions}"
            )
        
        if sex_mapped is None:
            raise HTTPException(
                status_code=422, 
                detail="Invalid sex. Must be 'male' or 'female'."
            )
        
        if smoker_mapped is None:
            raise HTTPException(
                status_code=422, 
                detail="Invalid smoker. Must be 'yes' or 'no'."
            )

        # --- 3. Create DataFrame ---
        # Create the DataFrame with the CORRECT mapped values
        df = pd.DataFrame([{
            "age": input_data.age,
            "sex": sex_mapped,         # <-- Use the mapped value
            "bmi": input_data.bmi,
            "children": input_data.children,
            "smoker": smoker_mapped,   # <-- Use the mapped value
            "region": region_raw
        }])

        # --- 4. Predict ---
        prediction = best_model_gb.predict(df)[0]
        
        # Return the successful prediction
        return {"predicted_charges": round(float(prediction), 2)}

    except Exception as e:
        # --- 5. Catch-all Error Handling ---
        # This stops the server from crashing with a 500 error
        # and tells the frontend *what* went wrong.
        print(f"Error during prediction: {e}") # Log for your own debugging
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)