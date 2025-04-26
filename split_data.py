import pandas as pd
from sklearn.model_selection import train_test_split

from load_data import load_city_data


# noinspection PyShadowingNames
def split_data():
    data = load_city_data(city="Philadelphia")
    business_topics= pd.read_csv('NLP/business.csv', encoding='utf-8')
    users_topics= pd.read_csv('NLP/users.csv', encoding='utf-8')
    review_df = data["reviews"]
    restaurant_df = data["businesses"]

    restaurant_df = pd.merge(
    restaurant_df,
    business_topics,
    how="inner",
    on="business_id" # Añade sufijos para distinguir columnas duplicadas
    ).reset_index(drop=True)



    review_df = (
        pd.merge(review_df, users_topics, how="inner", on="user_id")
        .reset_index(drop=True)
    )
    full_df = (
        pd.merge(restaurant_df, review_df, how="left", on="business_id",
        suffixes=('_x', '_y') )
        .sort_values(by="review_id")
        .reset_index(drop=True)
    )
    full_df.rename(
        columns={"stars_x": "restaurant_stars", "stars_y": "stars"}, inplace=True
    )
        # Buscar columnas duplicadas (las que aparecen con ambos sufijos)
    common_columns = set(col.replace('_x', '') for col in full_df.columns if col.endswith('_x'))
    common_columns &= set(col.replace('_y', '') for col in full_df.columns if col.endswith('_y'))

    # Restar los valores de columnas duplicadas
    for col in common_columns:
        full_df[col] = full_df[f'{col}_x'] - full_df[f'{col}_y']

        # Elimina las columnas originales con sufijos (opcional)
        full_df.drop([f'{col}_x', f'{col}_y'], axis=1, inplace=True)
    y = full_df["stars"].values
    full_df.drop(columns=["stars"], inplace=True)
    topic_cols = [f"topic_{i}" for i in range(6)]  # Asume que van del 0 al 5
    other_cols = [col for col in full_df.columns if col not in topic_cols]
    full_df = full_df[other_cols + topic_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        full_df, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()
    print(len(X_train.columns))
    print(X_train.columns)
