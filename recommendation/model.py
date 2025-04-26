import pandas as pd
import os

class Model:
    def __init__(self, trained_model):
        self.model = trained_model

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.df_user = pd.read_csv(os.path.join(BASE_DIR, '../NLP/users.csv'))
        self.df_business = pd.read_csv(os.path.join(BASE_DIR, 'business_transformed.csv'))

        # Identify topic columns (those starting with 'topic_')
        self.topic_columns = [col for col in self.df_business.columns if col.startswith('topic_')]

    def predict(self, user_id, business_id):
        # Get business row
        business_row = self.df_business[self.df_business['business_id'] == business_id]
        if business_row.empty:
            print(f"No business found with ID: {business_id}")
            return None  # or you could return a default value like np.nan

        # Get user row
        user_row = self.df_user[self.df_user['user_id'] == user_id]
        if user_row.empty:
            print(f"No user found with ID: {user_id}")
            return None  # or you could return a default value like np.nan

        # First, drop 'business_id'
        business_row_no_id = business_row.drop(columns=['business_id'])

        # Now, calculate the difference for the topic columns
        topics = [f'topic_{i}' for i in range(6)]  # ['topic_0', ..., 'topic_5']

        for topic in topics:
            business_row_no_id[topic] = business_row_no_id[topic].values - user_row[topic].values

        # Final result
        prediction = self.model.predict(business_row_no_id)

        return float(prediction)
