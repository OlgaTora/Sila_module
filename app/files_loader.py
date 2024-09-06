import os
import pandas as pd


def bypass_dir(folder_path: str, delimiter: str) -> pd.DataFrame | None:
    """Функция для обхода директории рекурсивно и объединения всех файлов csv в один датасет"""
    full_data = None
    buff = []
    for dir_path, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('csv'):
                df = file_read(os.path.join(dir_path, filename), delimiter)
                buff.append(df)
                full_data = pd.concat(buff, ignore_index=True)
    if not buff:
        print("Нет файлов с расширением .csv в указанной директории")
        return
    return full_data


def file_read(path: str, delimiter: str) -> pd.DataFrame | None:
    """Функция для чтения файлов csv"""
    try:
        df = pd.read_csv(path, delimiter=delimiter)
        return df
    except FileNotFoundError:
        print("Загрузите файл в директорию")


def determinate_file_or_dir(params: tuple) -> pd.DataFrame | None:
    """Функция определяющая файл или директория и обращающаяся к соответствующим функциям загрузки данных"""
    df = None
    path, delimiter = params
    if path.endswith('csv'):
        df = file_read(path, delimiter)
    elif os.path.isdir(path):
        df = bypass_dir(path, delimiter)
    else:
        print("Данные не загружены. Проверьте указанный путь.")
    return df
