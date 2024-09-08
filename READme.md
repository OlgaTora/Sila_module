Утилита командной строки для обучения и тестирования модели.

Порядок запуска:
1. Установить docker
2. При первом запуске:
docker build . -t sila
docker run -it --name sila sila
3. При повторных использованиях:
docker start -i sila
