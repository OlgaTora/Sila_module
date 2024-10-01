import pickle
import warnings
import pandas as pd

from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm

from SETTINGS import PERIOD
from data_transform import append_period_col, append_disc_type
from files_loader import determinate_file_or_dir

warnings.filterwarnings("ignore")


class ModelClassification:

    def __init__(self):
        self.df = None

    def __load_data(self, path_to_file, delimiter):
        """Функция для загрузки данных из пути, указанного при вызове сервиса train"""
        self.df = determinate_file_or_dir(path_to_file, delimiter)
        if self.df is not None:
            print('Данные загружены успешно')
        return self.df

    def __preprocessing(self, path_to_file, delimiter):
        """Добавление к данным колонки с количествои дней и типом диска"""
        self.df = self.__load_data(path_to_file, delimiter)
        self.df = append_period_col(self.df)
        self.df = append_disc_type(self.df)
        return self.df

    @staticmethod
    def get_preprocessor(X: pd.DataFrame):
        """Заполнение пропусков, нормализация и кодирование признаков"""
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

    def get_pipeline(self, X: pd.DataFrame, estimators: int, depth: int):
        """Пайплайн препроцессинга и построения модели"""
        preprocessor = self.get_preprocessor(X)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomSurvivalForest(
                max_depth=depth, n_estimators=estimators, random_state=1234, n_jobs=-1, min_samples_leaf=2,
                min_samples_split=5))
        ])

        return pipeline

    def __split(self):
        """
        Разделение данных на целевые переменные и признаки в соответствии с проведенным анализом данных.
        """
        y = self.df[['failure', 'days_between']]
        y['failure'] = y['failure'].astype('bool')
        X = self.df[
            [
                "smart_1_raw",
                "smart_9_raw",
                "smart_5_raw",
                "smart_12_raw",
                "smart_194_raw",
                "type"
            ]
        ]
        return X, y

    def save_trained_model(self, params):
        path_to_file, delimiter, number_of_estimators, maximum_depth = params
        self.df = self.__preprocessing(path_to_file, delimiter)
        X, y = self.__split()
        pipeline = self.get_pipeline(X, number_of_estimators, maximum_depth)
        y_surv = Surv.from_dataframe('failure', 'days_between', y)
        with tqdm(total=100, desc="Идет обучение модели") as pbar:
            model = pipeline.fit(X, y_surv)
            pbar.update(50)
            train_score = model.score(X, y_surv)
            pbar.update(50)
        model_pkl_file = "model.pkl"
        with open(model_pkl_file, 'wb') as file:
            pickle.dump(model, file)
        return train_score

    def get_prediction(self, params):
        """Получение предсказания в указанные периоды времени."""
        path_to_file, delimiter = params
        pkl_filename = "model.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        self.df = self.__preprocessing(path_to_file, delimiter)
        X_test, y_test = self.__split()
        pred = model.predict_survival_function(X_test)
        data = []
        for i in range(len(pred)):
            for t in PERIOD:
                data.append({
                    'index': i,
                    'Period': 'month' if t == 30 else ('quarter' if t == 90 else 'year'),
                    'Survival': 'true' if pred[i](t) > 0.5 else 'false'
                })
        res = pd.DataFrame(data)
        res_ = pd.DataFrame(self.df.serial_number).reset_index()
        result = pd.merge(res, res_, on='index')
        result.drop(columns='index', inplace=True)
        result.set_index('serial_number', inplace=True)
        return result
