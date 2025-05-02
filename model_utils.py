# model_utils.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

class Model_utils:
    def load_data(csv_path: str) -> pd.DataFrame:
        """Load your encoded dataframe from CSV."""
        return pd.read_csv(csv_path)

    def preprocess(
        df: pd.DataFrame,
        window_size,
        test_size=0.25, 
        random_state=42
    ):
        print(df.columns)
        poly = PolynomialFeatures(degree=2)

        X_raw = df.drop(columns=['crude_COVID_rate'])  # still a DataFrame

        poly.fit(X_raw)  # fit on the DataFrame so it knows column names
        X = poly.transform(X_raw)  # now X becomes a NumPy array
        poly_feature_names = poly.get_feature_names_out(X_raw.columns)
        windows = []
        unique_times = df['time'].unique()

        window_size = 2  # explicitly 2-month chunks

        for i in range(len(unique_times) - window_size + 1):  # still increments by 1 month
            start_time = unique_times[i]
            end_time = unique_times[i + 1]  # always span 2 months

            window_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            X_window = window_df.drop(columns=['crude_COVID_rate'])
            y_window = window_df['crude_COVID_rate']
            
            X_window_poly = poly.fit_transform(X_window)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_window_poly, y_window, test_size=test_size, random_state=random_state
            )
            
            windows.append((X_tr, X_te, y_tr, y_te))

        return windows, poly_feature_names

    def train_and_save(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_path: str,
        alpha: float = 0.1,
        max_iter: int = 10_000
    ):
        """
        Trains a Lasso(alpha, max_iter), saves it to model_path, and returns it.
        """
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_train, y_train)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        return model

    def load_model(model_path: str):
        """Loads and returns a joblib'ed model."""
        return joblib.load(model_path)


    def predict_new(
        df_new: pd.DataFrame,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        poly: PolynomialFeatures,
        model: Lasso
    ):
        """
        Given a new DataFrame with the same features (including 'time'),
        returns predictions on the original target scale.
        """
        # drop target if present
        if 'crude_COVID_rate' in df_new.columns:
            df_new = df_new.drop('crude_COVID_rate', axis=1)

        # standardize X
        X_std = pd.DataFrame(x_scaler.transform(df_new), columns=df_new.columns)


        # polynomial transform
        X_poly = poly.transform(X_std)

        # predict & invert scaling
        y_pred_std = model.predict(X_poly).reshape(-1, 1)
        y_pred     = y_scaler.inverse_transform(y_pred_std).ravel()

        return y_pred


    def evaluate(model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Prints and returns (mse, r2).
        """
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.4f} | R²: {r2:.4f}")
        return mse, r2
