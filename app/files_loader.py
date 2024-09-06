import os
import pandas as pd


def bypass_dir(folder_path, delimiter):
    global full_data
    buff = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('csv'):
                df = file_read(os.path.join(dirpath, filename), delimiter)
                buff.append(df)
                full_data = pd.concat(buff, ignore_index=True)
    if not buff:
        print("Нет файлов с расширением .csv в указанной директории")
        return
    return full_data


def file_read(path, delimiter):
    try:
        df = pd.read_csv(path, delimiter=delimiter)
        print('Data is load')
        return df
    except FileNotFoundError:
        print("Загрузите файл в директорию")


def determinate_file_or_dir(params):
    path, delimiter = params
    if path.endswith('csv'):
        df = file_read(path, delimiter)
        return df
    elif os.path.isdir(path):
        print('by')
        bypass_dir(path, delimiter)
        # if df is NoneL
