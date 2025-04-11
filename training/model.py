from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

class Model:
    def __init__(self, trained_model):
        self.model = trained_model

    def predict(self, user_id, business_id):
        # Load datasets
        df_business = pd.read_csv('../NLP/business.csv')
        df_user = pd.read_csv('../NLP/users.csv')
        # Get business row
        business_row = df_business[df_business['business_id'] == business_id]
        if business_row.empty:
            print(f"No business found with ID: {business_id}")
            business_row = None
        else:
            business_row = business_row.iloc[0]

        # Get user row
        user_row = df_user[df_user['user_id'] == user_id]
        if user_row.empty:
            print(f"No user found with ID: {user_id}")
            user_row = None
        else:
            user_row = user_row.iloc[0]
        if user_id is not None and user_row is not None:
            user_values = user_row.drop('user_id').values
            business_values = business_row.drop('business_id').values

            # Calculate the vector for the review
            distance = abs(user_values - business_values)
            return self.model.predict(distance)