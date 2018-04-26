FROM tensorflow/tensorflow

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /usr/src/app

EXPOSE 8081

ENV CUDA_VISIBLE_DEVICES ""

ENTRYPOINT ["python"]
CMD ["-m", "image_waiter_server"]