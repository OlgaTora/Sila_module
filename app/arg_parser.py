import argparse

from model import ModelClassification
from logger import logger


class ArgParser:
    def __init__(self):
        self.arguments = None
        self.parser = argparse.ArgumentParser(
            prog="check disk module",
            description="Утилита для предсказания по дискам,\
                                      установленным в серверах, для выдачи прогноза их выхода из строя.",
        )
        self.subparsers = self.parser.add_subparsers(help="sub-services help")

    def train_model(self):
        params = (self.arguments.path_to_file,
                  self.arguments.delimiter,
                  self.arguments.number_of_estimators,
                  self.arguments.maximum_depth)
        model = ModelClassification()
        score = model.save_trained_model(params)
        print(f"Модель обучена,\nТочность на тренировочной выборке={score}")

    def check_discs_health(self):
        params = self.arguments.path_to_file, self.arguments.delimiter
        model = ModelClassification()
        try:
            pred = model.get_prediction(params)
            print(f"Получен результат:\n{pred}")
        except ValueError:
            print("Файл не соответствует необходимому.")
        print("Задача выполнена.")

    def train_model_service(self):
        train_service = self.subparsers.add_parser(
            "train_model",
            help="Обучение модели",
            description="Обучение модели на данных из указанного файла или директории.",
        )
        train_service.set_defaults(func=self.train_model)

        train_service.add_argument(
            "-p",
            "--path_to_file",
            help="Путь к файлу или директории с данными для обучения модели",
            required=True,
            type=str,
        )

        train_service.add_argument(
            "-e",
            "--number_of_estimators",
            help="Количество деревьев в модели",
            required=True,
            type=int,
        )

        train_service.add_argument(
            "-t",
            "--maximum_depth",
            help="Глубина дерева в модели",
            required=True,
            type=int,
        )

        train_service.add_argument(
            "-d",
            "--delimiter",
            help="Разделитель для csv - файла. По умолчанию ,",
            default=",",
        )

    def check_disc_service(self):
        check_service = self.subparsers.add_parser(
            "check",
            help="Проверка дисков",
            description="Проверка одного или нескольких дисков на вероятность\
                                                    выхода из строя",
        )
        check_service.set_defaults(func=self.check_discs_health)

        check_service.add_argument(
            "-p",
            "--path_to_file",
            help="Путь к файлу или директории с данными для обучения модели",
            required=True,
            type=str,
        )
        check_service.add_argument(
            "-d",
            "--delimiter",
            help="Разделитель для csv - файла. По умолчанию ,",
            default=",",
        )

    def parse(self):
        """Функция для получения данных из cli с обработкой ошибок."""
        self.train_model_service()
        self.check_disc_service()
        try:
            self.arguments = self.parser.parse_args()
            if hasattr(self.arguments, "func"):
                self.arguments.func()
            else:
                self.parser.print_help()
        except SystemExit as e:
            if e.code != 0:
                log_msg = 'Проверьте количество аргументов, обратитесь к "--help"'
                print(log_msg)
        except Exception as e:
            log_msg = f"Произошла ошибка: {e}"
            print(log_msg)
