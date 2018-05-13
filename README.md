# faces
## Finding Celebrity Look-alikes Using NMSW Index and Facenet Embeddings

Сайт: http://93.175.11.8:8080

### Промежуточный отчет

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

### Финальный отчет

Реализован свой индекс на основе алгоритма Hierarchical Navigable Small World, покрыт тестом на корректность (на 10к векторов размерности 300 результат идентичен наивному линейному индексу). Распараллеливание при помощи OpenMP сильно затруднено из-за множества общих ресурсов (множество вложенных друг в друга объектов, представляющих собой граф), поэтому от него было решено отказаться.   
Производительность индекса сравнена на Facenet эмбеддингах с `nmslib` и с алгоритмами `BallTree` и `KDTree` из `sklearn`: запросы к индексу выполняются быстрее чем в sklearn, но nmslib индекс проигрывает, скорее всего это связанно с оптимизациями вроде префетча потенциальных соседей в память по мере выполнения запроса.
![](https://i.imgur.com/mXQN7Ra.png)
![](https://i.imgur.com/w6KLgh0.png)

Оценка качества запросов.
Качество запросов приведено в таблицах ниже.
Размеры на обучении: 100000, на тестировании: 100.
(Обучение - построение индекса, тестирование - поиск ближайших элементов, усреднение)

Выбрали следующие метрики (все значения метрик усредняются по тестовой выборке - среднее по 100):

1) chosen position: на каком месте находится ближайший найденный индексом сосед
2) in 10 best: какое количество из первых 10 найденных индексом попадают в десятку истинно ближайших

Измерения проводились для метрики эвклида. по возможности с сохранением остальных параметров стандартными.
Ниже в таблицах KDTree, по сути, представляет эталон метрик.
k - количество ближайших соседей, которые ищем.


| k = 1       | chosen position | in 10 best |  dcg  |
|-------------|-----------------|------------|-------|
| KDTree      | 0.0             | 1.0        | 1.000 |
| Annoy       | 195.99          | 0.1        | 0.042 |
| NMSLib      | 7.19            | 0.77       | 0.320 |
| HNSW Custom | 0.34            | 1.0        | 0.875 |

| k = 5       | chosen position | in 10 best |  dcg  |
|-------------|-----------------|------------|-------|
| KDTree      | 0.0             | 5.0        | 1.667 |
| Annoy       | 195.99          | 0.1        | 0.050 |
| NMSLib      | 7.19            | 1.5        | 0.453 |
| HNSW Custom | 0.33            | 4.98       | 1.410 |

| k = 10      | chosen position | in 10 best |  dcg  |
|-------------|-----------------|------------|-------|
| KDTree      | 0.0             | 10.0       | 1.876 |
| Annoy       | 130.53          | 0.16       | 0.083 |
| NMSLib      | 7.19            | 1.51       | 0.483 |
| HNSW Custom | 0.32            | 7.07       | 1.579 |

| k = 100     | chosen position | in 10 best |  dcg  |
|-------------|-----------------|------------|-------|
| KDTree      | 0.0             | 10.0       | 2.336 |
| Annoy       | 14.92           | 0.72       | 0.333 |
| NMSLib      | 7.19            | 1.51       | 0.526 |
| HNSW Custom | 0.11            | 8.33       | 2.094 |


Видим, что реализованный индекс из коробки работает лучше рассмотренных выше (конечно за исключением KDTree), хоть и проигрывает по скорости.

Сервис по поиску знаменитостей разбит на два микросервиса:
1) image_waiter_server — принимает на вход картинку в веб-интерфейсе и конвертирует ее в эмбеддинг
2) index_search_server — микросервис с индексом. Принимает эмбеддинг, выдает индексы ближайших соседей
Сайт запускается в docker контейнерах (docker-compose)

Структурно:

image_waiter_server содержит:
1) натренированную модель facenet
2) изображения знаменитостей, сопоставленные с некоторыми индексами

index_search_server содержит:
1) модель индекса. в данный момент есть поддержка annoy и hnsw custom, сейчас используется hnsw custom

эмбединги получены из преобразованных/обрезанных фото знаменитостей. Преобразование фото для получения эмбеддинга проводилось тем же способом, что и используется в сервисе.

Процесс:
1) Изображение попадает на image_waiter_server
2) Поиск лица на изображении, обрезание, скалирование изображения
3) Извлечение эмбеддинга с помощью facenet модели
4) Получение от index_search_server сервиса номеров ближайших соседей
5) Выдача изображений, соответствующих полученным номерам

Готовые эмбеддинги и модели индексов: https://yadi.sk/d/GWOOdryL3UDcqK (annoy, nmslib, hnsw custom)

Интерфейс сервиса и примеры результатов: https://github.com/strawberrypie/faces/blob/master/Interface.md

Типичные времена запроса:
1) Обрезание и выравнивание изображения (image alignment): 0.02 - 0.20s
2) Получение вектора изображения (embedding): 0.05 - 0.10s
3) Выполнение запроса на поиск ближайших соседей (index): 0.007 - 0.008s
