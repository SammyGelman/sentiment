FROM tensorflow/tensorflow:nightly-gpu

RUN apt update
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install gunicorn
RUN pip3 install pandas
<<<<<<< HEAD
RUN pip3 install flask
RUN pip3 install Flask-SQLAlchemy
RUN pip3 install keras 
RUN pip3 install protobuf==3.20.*
=======
RUN pip3 install protobuf==3.20.*

COPY . .

EXPOSE 3000
>>>>>>> be55347af9e637877753df0ea65ad70ea80e55f9
