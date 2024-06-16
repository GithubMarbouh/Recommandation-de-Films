import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Add, Dense
from tensorflow.keras.optimizers import Adam
from titleApp.models import Myrating


def get_data():
    df = pd.DataFrame(list(Myrating.objects.all().values()))
    user_ids = df.user_id.unique()
    movie_ids = df.movie_id.unique()
    user_id_map = {id: i for i, id in enumerate(user_ids)}
    movie_id_map = {id: i for i, id in enumerate(movie_ids)}
    df['user_id'] = df['user_id'].map(user_id_map)
    df['movie_id'] = df['movie_id'].map(movie_id_map)
    return df, len(user_ids), len(movie_ids)


def create_model(num_users, num_movies, embedding_size=50):
    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
    movie_vec = Flatten()(movie_embedding)

    dot_product = Dot(axes=1)([user_vec, movie_vec])

    model = Model([user_input, movie_input], dot_product)
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    return model


def train_model():
    df, num_users, num_movies = get_data()
    X = df[['user_id', 'movie_id']].values
    y = df['rating'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(num_users, num_movies)
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, validation_data=([X_val[:, 0], X_val[:, 1]], y_val))

    return model


def make_predictions(user_id, movie_id, model):
    user_data = np.array([user_id])
    movie_data = np.array([movie_id])
    prediction = model.predict([user_data, movie_data])
    return prediction[0][0]


# Example usage
update_ratings()
model = train_model()
prediction = make_predictions(user_id=1, movie_id=1, model=model)
print(f"Predicted rating for user 1 and movie 1: {prediction}")

