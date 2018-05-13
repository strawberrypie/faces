FROM python:3

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/strawberrypie/faces.git # bad, but what can we do now :(
RUN pip3 install -e faces/index/hnsw

COPY . /usr/src/app

ENTRYPOINT ["python3"]

CMD ["-m", "index_search_server"]