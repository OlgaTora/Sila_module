import pickle
import warnings

from sklearn.impute import SimpleImputer
from sksurv.util import Surv

from data_transform import append_period_col

warnings.filterwarnings("ignore")
from files_loader import determinate_file_or_dir
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest


class ModelClassification:

    def __init__(self):
        self.df = None

    def __load_data(self, params):
        print(params)
        """Функция для загрузки данных из пути, указанного при вызове сервиса train"""
        self.df = determinate_file_or_dir(params)
        if self.df is not None:
            print('Данные загружены')
        return self.df

    def __preprocessing(self, params):
        self.df = self.__load_data(params)
        self.df = append_period_col(self.df)
        threshold = len(self.df) * 0.5
        self.df = self.df.dropna(axis=1, thresh=threshold)
        # print(self.df.columns)
        self.df = self.df.loc[:, self.df.nunique() > 1]
        # print(self.df.columns)
        # удалить корреоирующие столбцы еще
        self.df.drop(columns=['serial_number', 'date'], axis=1, inplace=True)
        return self.df

    @staticmethod
    def get_preprocessor(X: pd.DataFrame):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler())
                ]), X.select_dtypes(include="number").columns),

                ('cat', Pipeline(steps=[
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), X.select_dtypes(exclude="number").columns)
            ])
        return preprocessor

    def get_pipeline(self, X: pd.DataFrame):
        preprocessor = self.get_preprocessor(X)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomSurvivalForest(n_estimators=50, random_state=42))
        ])

        return pipeline

    def __split(self):
        y = self.df[['failure', 'days_between']]
        y['failure'] = y['failure'].astype('bool')
        X = self.df.drop(columns=['failure', 'days_between'])
        return X, y

    def save_trained_model(self, params):
        self.df = self.__preprocessing(params)
        print(self.df.columns)
        X, y = self.__split()
        pipeline = self.get_pipeline(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train_surv = Surv.from_dataframe('failure', 'days_between', y_train)
        # y_test_surv = Surv.from_dataframe('failure', 'days_between', y_test)

        # Обучение модели
        model = pipeline.fit(X_train, y_train_surv)
        # model.fit(X, y)
        model_pkl_file = "model.pkl"
        with open(model_pkl_file, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def get_prediction(X_test):
        pkl_filename = "model.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict(X_test)
        return prediction[0]
