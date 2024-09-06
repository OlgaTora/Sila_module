from sklearn import linear_model
import pickle
from sklearn.datasets import load_iris
import pandas as pd


# try:
#     df = pd.read_csv("_______", delimiter=",")
# except FileNotFoundError:
#     print("You have to load the file to the directory before opening it.")

class ModelClassification:

    def __init__(self):
        self.df = None

    def load_data_(self):
        # тут загрузка данных из консоли
        self.df = load_iris(as_frame=True).frame
        return self.df

    def preprocessing_(self):
        self.df = self.load_data_()
        self.df['target'] = self.df.target.apply(lambda x: 1 if x == 2 else 0)
        return self.df

    def split_(self):
        X = self.df.drop(["target"], axis=1)
        y = self.df["target"]
        return X, y

    @staticmethod
    def init_model_():
        model = linear_model.LogisticRegression(
            solver="liblinear", random_state=123, class_weight="balanced"
        )
        return model

    def save_trained_model(self):
        self.df = self.preprocessing_()
        X, y = self.split_()
        model = self.init_model_()
        model.fit(X, y)
        model_pkl_file = "model.pkl"
        with open(model_pkl_file, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def get_prediction(disc_params):
        pkl_filename = "model.pkl"
        X_test = pd.DataFrame(disc_params).T
        X_test.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                          'petal width (cm)']
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict(X_test)
        return prediction[0]
