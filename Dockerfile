FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt /hair-segmentation/requirements.txt
WORKDIR hair-segmentation
COPY . /hair-segmentation
RUN pip install -r requirements.txt
CMD ["python", "train.py"]