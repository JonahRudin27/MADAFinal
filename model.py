from model_utils import Model_utils
import matplotlib.pyplot as plt
import pandas as pd
from model_utils import Model_utils
import joblib
import numpy as np


class Model:
    def __init__(self, train = False):
        df = Model_utils.load_data('df_encoded.csv')

        self.windows, self.poly_feature_names = Model_utils.preprocess(df, 1)

        if train == True:
            print("retraining")
            self.models = [
                Model_utils.train_and_save(w[0], w[2], f'models/model_month_{i}.pkl')
                for i, w in enumerate(self.windows)
            ]
        else:
            print("loading trained model")
            self.models = [
                joblib.load(f'models/model_month_{i}.pkl')
                for i in range(len(self.windows))
            ]
    
    def test_models(self):
        """Load models with timestamps from pickle files"""
        for i in range(len(self.models)):
            Model_utils.evaluate(self.models[i],self.windows[i][1], self.windows[i][3])

    def graph_feature(self, feature_name):
        coefficients = []

        for model in self.models:
            if not hasattr(model, 'coef_'):
                coefficients.append(np.nan)
                continue

            # Use the shared feature name list
            feature_index = np.where(self.poly_feature_names == feature_name)[0]
            if feature_index.size == 0:
                coefficients.append(np.nan)
            else:
                coef = model.coef_.flatten()[feature_index[0]]
                coefficients.append(coef)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(coefficients)), coefficients, 
                marker='o', linestyle='--', color='tab:blue')
        plt.title(f'Coefficient Evolution: {feature_name}', pad=20)
        plt.xlabel('Time Window Index', labelpad=15)
        plt.ylabel('Coefficient Value', labelpad=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
