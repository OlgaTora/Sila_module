import gc
import os
import pandas as pd

from tqdm import tqdm
from data_transform import reduce_data


def bypass_dir(folder_path: str, delimiter: str) -> pd.DataFrame | None:
    """Функция для обхода директории рекурсивно и объединения всех файлов csv в один датасет"""
    full_data = None
    buff = []
    for dir_path, _, filenames in tqdm(os.walk(folder_path)):
        for filename in filenames:
            if filename.endswith("csv"):
                df = file_read(os.path.join(dir_path, filename), delimiter)
                buff.append(df)
    if not buff:
        print("Нет файлов с расширением .csv в указанной директории")
        return
    full_data = pd.concat(buff, ignore_index=True)
    return full_data


def file_read(path: str, delimiter: str) -> pd.DataFrame | None:
    """Функция для чтения файлов csv"""
    try:
        gc.collect()
        df = pd.read_csv(path, delimiter=delimiter, encoding="unicode_escape")
        df = reduce_data(df)
        df = pd.concat([df.loc[df.failure == 1], df.sample(10)], ignore_index=True)
        return df
    except FileNotFoundError:
        print("Загрузите файл в директорию")


def determinate_file_or_dir(path: str, delimiter: str) -> pd.DataFrame | None:
    """Функция определяющая файл или директория и обращающаяся к соответствующим функциям загрузки данных"""
    df = None
    # path, delimiter = params
    if path.endswith("csv"):
        df = file_read(path, delimiter)
    elif os.path.isdir(path):
        df = bypass_dir(path, delimiter)
    else:
        print("Данные не загружены. Проверьте указанный путь.")
    return df


# determinate_file_or_dir(('/home/olgatorres/PycharmProjects/Sila_module/app/test_data/2.csv', ','))
