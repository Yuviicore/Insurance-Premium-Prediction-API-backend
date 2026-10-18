
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from model.predict import model_predict, model, MODEL_VERSION
from schema.user_input import userInput

app = FastAPI(title="Insurance Premium Prediction API")


@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}


@app.get("/health")
def health():
    return {
        "status": 200,
        "version": MODEL_VERSION,
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: userInput):
    user_input = {
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }

    try:
        prediction = model_predict(user_input)
        return {"predicted_category": str(prediction)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

