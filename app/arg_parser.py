import argparse

from files_loader import get_X_test_from_params, file_read
from model import ModelClassification
from logger import logger


class ArgParser:
    def __init__(self):
        self.arguments = None
        self.parser = argparse.ArgumentParser(prog='check disk module',
                                              description="Утилита для предсказания по дискам,\
                                      установленным в серверах, для выдачи прогноза их выхода из строя.")
        self.subparsers = self.parser.add_subparsers(help='sub-services help')

    def train_model(self):
        params = self.arguments.path_to_file, self.arguments.delimeter
        model = ModelClassification()
        model.save_trained_model(params)
        print("Model've trained")

    def check_disc_health(self):
        params = (self.arguments.param_a,
                  self.arguments.param_b,
                  self.arguments.param_c,
                  self.arguments.param_d)
        model = ModelClassification()
        data = get_X_test_from_params(params)
        pred = model.get_prediction(data)
        print('Disk is good') if pred == 1 else print('Disk is not healthy')
        print("Disk health done")

    def check_discs_health(self):
        params = self.arguments.path_to_file, self.arguments.delimeter
        model = ModelClassification()
        data = file_read(*params)
        try:
            pred = model.get_prediction(data)
            print(pred)
        except ValueError:
            print('Файл не соответствует необходимому')
        print("Disk health done")

    def train_model_service(self):
        train_service = self.subparsers.add_parser('train_model', help='Обучение модели',
                                                   description='Обучение модели на данных из указанного файла или директории')
        train_service.set_defaults(func=self.train_model)
        # arguments for train_model service
        train_service.add_argument('-p', '--path_to_file',
                                   help='Путь к файлу или директории с данными для обучения модели',
                                   required=True,
                                   type=str)
        train_service.add_argument('-d', '--delimeter',
                                   help='Разделитель для csv - файла. По умолчанию ;',
                                   default=';'
                                   )

    def check_disc_service(self):
        check_service = self.subparsers.add_parser('check', help='Проверка дисков',
                                                   description='Проверка одного или нескольких дисков на вероятность\
                                                    выхода из строя')
        sub_subparsers = check_service.add_subparsers(help='sub-services help')

        check_one_disc = sub_subparsers.add_parser('one_disc', help='Проверка одного диска',
                                                   description='Проверка одного диска на вероятность выхода из строя,\
                                                    ввод параметров из cli')
        check_one_disc.set_defaults(func=self.check_disc_health)
        # arguments for check one disc
        check_one_disc.add_argument('-a', '--param_a',
                                    help='Disc parameters',
                                    required=True,
                                    type=int)
        check_one_disc.add_argument('-b', '--param_b',
                                    help='Disc parameters',
                                    required=True,
                                    type=int)
        check_one_disc.add_argument('-c', '--param_c',
                                    help='Disc parameters',
                                    required=True,
                                    type=int)
        check_one_disc.add_argument('-d', '--param_d',
                                    help='Disc parameters',
                                    required=True,
                                    type=int)

        check_discs = sub_subparsers.add_parser('discs', help='Проверка нескольких дисков',
                                                description='Проверка дисков, загруженных в виде файла .csv,\
                                                 на вероятность выхода из строя')
        check_discs.set_defaults(func=self.check_discs_health)

        # arguments for check discs from file
        check_discs.add_argument('-p', '--path_to_file',
                                 help='Путь к файлу или директории с данными для обучения модели',
                                 required=True,
                                 type=str)
        check_discs.add_argument('-d', '--delimeter',
                                 help='Разделитель для csv - файла. По умолчанию ;',
                                 default=';'
                                 )

    def parse(self):
        self.train_model_service()
        self.check_disc_service()
        try:
            self.arguments = self.parser.parse_args()
            self.arguments.func()
        except SystemExit as e:
            log_msg = 'Проверьте количество аргументов, обратитесь к "--help"'
            print(log_msg)
            logger.error(f'{e}: {log_msg}')
