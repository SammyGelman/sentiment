FROM tensorflow/tensorflow:nightly-gpu

RUN apt update
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install gunicorn
RUN pip3 install pandas
RUN pip3 install flask
RUN pip3 install Flask-SQLAlchemy
RUN pip3 install keras 
RUN pip3 install protobuf==3.20.*
