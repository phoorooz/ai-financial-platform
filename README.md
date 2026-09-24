# AI Financial Forecasting Platform



An end-to-end Machine Learning project built by **Ali Forouzan** for predicting Bitcoin's next-day return.



I built this project to practice the complete ML engineering workflow — from collecting and preparing data to training a model and serving predictions through an API.



\## 🚀 What I Built



```text

Yahoo Finance

    ↓

Data Cleaning \& Validation

    ↓

Feature Engineering

    ↓

Time-Series Train/Test Split

    ↓

XGBoost Model

    ↓

MLflow Experiment Tracking

    ↓

FastAPI

    ↓

Docker + Tests + CI

```



### Main technologies



* Python

* Pandas / NumPy

* XGBoost

* Scikit-learn

* MLflow

* FastAPI

* Docker

* Pytest

* GitHub Actions



## 📁 Project Structure



```text

ai-financial-platform/

├── app/              # FastAPI application

├── ml/

│   ├── data/         # Data pipeline

│   ├── features/     # Feature engineering

│   ├── training/     # Model training

│   ├── evaluation/   # Model evaluation

│   └── inference/    # Model inference

├── tests/            # Automated tests

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

```



## ▶️ Run the Project



### 1. Clone the repository



```bash

git clone https://github.com/phoorooz/ai-financial-platform.git

cd ai-financial-platform

```



### 2. Start Docker



```bash

docker compose build

docker compose up -d

```



### 3. Run the ML pipeline



```bash

docker compose exec python python ml/data/download_data.py

docker compose exec python python ml/data/prepare_data.py

docker compose exec python python ml/features/build_features.py

docker compose exec python python ml/training/split_data.py

docker compose exec python python ml/training/train_xgboost.py

docker compose exec python python ml/evaluation/evaluate_xgboost.py

```



### 4. Start the API



```bash

docker compose exec python uvicorn app.api.main:app --host 0.0.0.0 --port 8000

```



Then open the **Swagger API documentation** at:



`/docs`



The main endpoints are:



* `GET /health`

* `POST /predict`

* `GET /predict/latest`



## 📊 Current Model Results



The current XGBoost model achieved:



| Metric               |   Result |

| -------------------- | -------: |

| MAE                  | 0.025049 |

| RMSE                 | 0.031589 |

| Directional Accuracy |   49.88% |



These results are from the current experiment and are mainly used to evaluate and improve the ML pipeline.



## 🧪 MLflow



I use \*\*MLflow\*\* to track experiments, model parameters, metrics, and trained models.



This makes it easier to compare different experiments as the project evolves.



## 🧑‍💻 About



**Built by Ali Forouzan**



AI & Software Engineer interested in Machine Learning, AI, backend development, and practical ML systems.



GitHub: `phoorooz`



---



> **Note:** This is an educational ML engineering project, not a financial or trading recommendation.



