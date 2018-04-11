# faces

Реализован pipeline: find_sim_celebrities_debug.py

Pipeline уже готов работать с annoy/nmslib/собственной реализацией индекса. Полезно для будущих экспериментов.
Сейчас используется обученная модель Facenet.

Поиск похожего лица выполняется по датасету CelebA.

Изображения с CelebA предобрабатываются crop+scale для использования с Facenet моделью.

Полученные к текущему моменту эмбеддинги доступны по ссылке: https://yadi.sk/d/GWOOdryL3UDcqK.

Помио этого, там содержится текущий Annoy индекс для по эмбеддингам.

Индекс взят готовым: annoy/nmslib. Реализовываем HNSW. Текущая версия индекса однопоточная, планируется покрыть реализацию тестами, профилировщиком выяснить узкие места и ускорить их с помощью OpenMP.

Планируемая архитектура:

- Планируется docker compose. Первый сервис отвечает за отображение картинок, второй за поиск в индексе.
Первый сервис получает изображение, выделяет лицо (crop+scale), вычисляет embedding и передает его на второй сервис.
Второй сервис получает embedding и возвращает индексы ближайших элементов.
Первый сервис берет соответствующие полученным индексам пути и отображает ближайшие картинки.
- Изначально планировали хранить маппинг картинки-ембединги на втором сервисе, но, кажется, это плохая идея. Инициализация индекса может происходить и извне для своей задачи.
- Помимо этого, планируем реализовать свою нейросеть на quadruplets.
