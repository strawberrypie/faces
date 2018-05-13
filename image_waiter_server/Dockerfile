FROM python:3

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /usr/src/app

ENV CUDA_VISIBLE_DEVICES ""

ENTRYPOINT ["python3"]
CMD ["-m", "image_waiter_server"]