import pandas as pd
import numpy as np
import sklearn
import xgboost


def main():
    print("Pandas:", pd.__version__)
    print("NumPy:", np.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("XGBoost:", xgboost.__version__)
    print("\nML environment is ready!")


if __name__ == "__main__":
    main()