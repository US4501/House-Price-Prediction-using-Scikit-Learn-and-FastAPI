# House Price Prediction using Scikit-Learn and FastAPI

A machine learning project that predicts California house prices with a Random Forest model and serves predictions through a FastAPI REST API.

## Features

- Data exploration of the California Housing dataset
- Random Forest Regressor trained with scikit-learn
- Model evaluation using MAE and R² score
- Trained model and feature list saved with `joblib`
- FastAPI app with validated inputs (Pydantic) and CSV file upload support
- Interactive API docs (Swagger UI) built in

## Tech Stack

Python, scikit-learn, pandas, FastAPI, Pydantic, joblib, Uvicorn

## Project Structure

```
.
├── explore.py               # Explore the dataset (shape, head, summary stats)
├── train.py                 # Train the model and save it with joblib
├── main.py                  # FastAPI app that serves predictions
├── house_model_joblib       # Trained model (tracked with Git LFS)
├── house_features_joblib    # Feature names used by the model
├── .gitattributes           # Git LFS configuration
└── .gitignore
```

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn, loaded with `fetch_california_housing()`.

| Feature | Description |
|---|---|
| `MedInc` | Median income in the neighbourhood |
| `HouseAge` | Average age of houses in the block |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block population |
| `AveOccup` | Average occupants per household |
| `Latitude` | Block latitude |
| `Longitude` | Block longitude |

The target is the median house value (in units of $100,000).

## Model

- Algorithm: `RandomForestRegressor`
- `n_estimators=100`, `random_state=42`
- Train/test split: 80% / 20%
- Metrics: Mean Absolute Error (MAE) and R² score

## Setup

```bash
git clone https://github.com/US4501/House-Price-Prediction-using-Scikit-Learn-and-FastAPI.git
cd House-Price-Prediction-using-Scikit-Learn-and-FastAPI

# Model files are stored with Git LFS
git lfs install
git lfs pull

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Usage

**1. (Optional) Retrain the model**

```bash
python train.py
```

**2. Start the API**

```bash
uvicorn main:app --reload
```

**3. Open the interactive docs**

Go to http://127.0.0.1:8000/docs to see all endpoints and try them out.

### Example input

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Compare other models (Gradient Boosting, XGBoost)
- Dockerize the API
- Deploy to a cloud platform

## Author

**Utkarsh Singh** — [@US4501](https://github.com/US4501)
