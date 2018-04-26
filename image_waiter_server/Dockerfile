FROM python:3

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip3 install --upgrade pip
RUN pip3 install requests
RUN pip3 install Flask
RUN pip3 install opencv_python
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install tensorflow
RUN pip3 install facenet
# RUN pip3 install -r requirements.txt

COPY . /usr/src/app

EXPOSE 8081

ENV CUDA_VISIBLE_DEVICES ""

ENTRYPOINT ["python"]
CMD ["-m", "image_waiter_server"]