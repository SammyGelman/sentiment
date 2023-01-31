FROM tensorflow/tensorflow:nightly-gpu
RUN apt update
RUN pip install matplotlib
RUN pip install --upgrade tensorflow-probability
RUN pip install numpy
RUN pip install scipy
RUN pip install tensorflow_datasets 
RUN pip install tensorflow_probability
RUN pip install pandas
RUN pip install keras 
