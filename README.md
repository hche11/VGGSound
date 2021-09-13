# VGGSound

Code and results for ICASSP2020 "VGGSound: A Large-scale Audio-Visual Dataset".

The repo contains the dataset file and our best audio classification model. 

## Dataset

To download VGGSound, we provide a [csv file](./data/vggsound.csv). For each YouTube video, we provide YouTube URLs, time stamps, audio labels and train/test split. Each line in the csv file has columns defined by here.

```
# YouTube ID, start seconds, label,train/test split. 
```

A helpful link for [data download](https://github.com/speedyseal/audiosetdl)!

## Audio classification 

We detail the audio classfication results here. 

* **Pretrain** refers whether the model was pretrained on [YouTube-8M dataset](https://github.com/tensorflow/models/tree/master/research/audioset/vggish). 
* **Dataset (common)** means it is a subset of the dataset. This subset only contains data of common classes ([listed here](./data/Common.txt)) between AudioSet and VGGSound. 
* **[ASTest](./data/AStest.csv)** is the intersection of AudioSet and VGGSound testsets.

| 	  | Model    | Aggregation   | Pretrain           | Finetune/Train  | Test          | mAP   | AUC   | d-prime |
|:---:|:--------:|:-------------:| :-------------:    |:--------------: |:-------------:|:-----:|:-----:|:-------:| 
| A   | VGGish   | \             | :heavy_check_mark: |AudioSet (common)| ASTest        | 0.286 | 0.899 | 1.803   |
| B   | VGGish   | \             | :heavy_check_mark: |VGGSound (common)| ASTest        | 0.326 | 0.916 | 1.950   | 
| C   | VGGish   | \             | :x:                |VGGSound (common)| ASTest        | 0.301 | 0.910 | 1.900   |
| D   | ResNet18 | AveragePool   | :x:                |VGGSound (common)| ASTest        | 0.328 | 0.923 | 2.024   |
| E   | ResNet18 | NetVLAD       | :x:                |VGGSound (common)| ASTest        | 0.369 | 0.927 | 2.058   |
| F   | ResNet18 | AveragePool   | :x:                |VGGSound         | ASTest        | 0.404 | 0.944 | 2.253   |
| G   | ResNet18 | NetVLAD       | :x:                |VGGSound         | ASTest        | 0.434 | 0.950 | 2.327   |
| H   | ResNet18 | AveragePool   | :x:                |VGGSound         | VGGSound      | 0.516 | 0.968 | 2.627   |
| I   | ResNet18 | NetVLAD       | :x:                |VGGSound         | VGGSound      | 0.512 | 0.970 | 2.660   |
| J   | ResNet34 | AveragePool   | :x:                |VGGSound         | ASTest        | 0.409 | 0.947 | 2.292   |
| K   | ResNet34 | AveragePool   | :x:                |VGGSound         | VGGSound      | 0.529 | 0.972 | 2.703   |
| L   | ResNet50 | AveragePool   | :x:                |VGGSound         | ASTest        | 0.412 | 0.949 | 2.309   |
| M   | ResNet50 | AveragePool   | :x:                |VGGSound         | VGGSound      | 0.532 | 0.973 | 2.735   |


## Environment

* Python 3.6.8
* Pytorch 1.3.0


## Pretrained model and evaluation 

We provide the pretrained models **H** an **I** here,
```
wget http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar
wget http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/I.pth.tar
```

To test the model and generate prediction files,

```
python test.py --data_path "directory to audios/" --result_path "directory to predictions/" --summaries "path to pretrained models" --pool "avgpool"
```


To evaluate the model performance using the generated prediction files,

```
python eval.py --result_path "directory to predictions/"
```

## Citation
```
@InProceedings{Chen20,
  author       = "Honglie Chen and Weidi Xie and Andrea Vedaldi and Andrew Zisserman",
  title        = "VGGSound: A Large-scale Audio-Visual Dataset",
  booktitle    = "International Conference on Acoustics, Speech, and Signal Processing (ICASSP)",
  year         = "2020",
}
```

## License
The VGG-Sound dataset is available to download for commercial/research purposes under a Creative Commons Attribution 4.0 International License. The copyright remains with the original owners of the video. A complete version of the license can be found [here](./LICENCE.txt).
