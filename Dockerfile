FROM tensorflow/tensorflow:nightly-gpu

RUN apt update
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install protobuf==3.20.*

COPY . .

EXPOSE 3000
