###Hair Segmentation

Hair segmentation using tensorflow 2.0 and tf.keras.

Model architecture from [https://arxiv.org/pdf/1712.07168.pdf]()

####Data
CelebAMask-HQ Dataset

* utils/data_prep.ipynb notebook for cleaning and preparing data

After data_prep: data will be split to train/valid/test folders.
Each folder will have /images and /masks subfolders. 
####Models

* MobileNet

####Configuration
Config files are in  folder /configs 

* train_config
```
{
  "path_to_dataset": "/path/to/hair-segmentation/dataset/",
  "checkpoint": true,  # saving checkpoints while training
  "img_size": 224,     # img width & height
  "logs": true,        # saving tensorboard logs
  "batch_size" : 16,
  "num_epochs" : 30,
  "model_prefix": "hair_segmentation_"
}
```

* test_config
```
{
  "path_to_dataset": "/path/to/hair-segmentation/dataset/test/",
  "results_folder": "test_results/",
  "img_size": 224,
  "model_path": "/path/to/trained/model.h5",
  "color": [255, 102, 255], # [b,g,r] color to apply to masked region
  "alpha": 0.8                  
}
```
####Usage

With tensorflow docker:

prerequisite: tensorflow-gpu docker image [https://www.tensorflow.org/install/docker]()

```console
docker build -t hair-segmentation .
docker run --gpus all -v /path/to/dataset:/hair-segmentation/dataset/
```

With tensorflow:

prerequisite: tensorflow-gpu 2.0
```console
pip install -r requirements.txt
python train.py
```

#### Demo
```
python demo.py
```

#### Demo results:

