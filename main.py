import joblib
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
import pandas as pd
import io
from pathlib import Path


app=FastAPI(title="California House Price API")

# load data 
BASE = Path(__file__).parent
model = joblib.load(BASE / "house_model_joblib")
features = joblib.load(BASE / "house_features_joblib")


# check is loaded or not
print(model)

# input schema
# ---------- Input schema (BaseModel = validates data sent to server) ----------
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in neighbourhood")
    HouseAge: float = Field(gt=0, description="Average age of houses in the block")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Population in the block")
    AveOccup: float = Field(gt=0, description="Average occupancy")
    Latitude: float = Field(ge=32, le=42, description="Latitude of block")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude of block")


# ---------- Routes ----------
@app.get("/")
def home():
    return {
        "message": "California house prediction API",
        "status": "Running",
        "endpoint": "Send POST request to /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "Running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000",
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        # Same column order the model was trained with
        input_data = pd.DataFrame([house.model_dump()])[features]

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousands",
            "confidence_range": f"${price_usd - 32773:,.0f} to ${price_usd + 32773:,.0f}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    # File(...) = value comes from the uploaded file, not the URL.
    # async = server can do other work while waiting for the upload.
    # Missing file -> FastAPI returns 422 Validation Error.
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file only")

    try:
        contents = await file.read()  # raw bytes
        df = pd.read_csv(io.BytesIO(contents))  # BytesIO = fake file in memory
       # ---------- Bulk upload ----------
        REQUIRED_COLUMNS = list(HouseFeatures.model_fields.keys())
        missing_columns = [c for c in REQUIRED_COLUMNS
        if c not in df.columns]
        if missing_columns:
         raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_columns}",
            )
        #  """model_fields → contains information about all model fields.
        #  .keys() → gets just their names.
        #   list(...) → converts those names into a normal Python list"""

        if len(df) == 0:
         raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data rows")

        predictions = model.predict(df[features])
        df["predicted_price_usd"] = (predictions * 100000).round(0)
        df["predicted_price_usd"] = df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)  # string; index=False -> no row numbers

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
  # keep the 400s above instead of turning them into 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")