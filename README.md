# VGGSound

Code for ICASSP2020 "VGGSound: A Large-scale Audio-Visual Dataset".

The repo contains the dataset file and our best audio classification model. 

## Dataset

To download VGGSound, we provide a csv file. For each YouTube video, we provide YouTube URLs, time stamps, audio labels and train/test split. Each line in the csv file has columns defined by here: # YouTube ID, start seconds, label,train/test split. The dataset can be found here. 

## Audio classification 

| 	  | Model    | Aggregation   | Pretrain           | Finetune/Train | Test          | mAP           | AUC            | d-prime       |
|:---:|:--------:|:-------------:| :-------------:    |:--------------:|:-------------:|:-------------:|:--------------:|:-------------:| 
| A   | VGGish   | \             | :heavy_check_mark: |
| B   | VGGish   | \             | :heavy_check_mark: |
| C   | VGGish   | \             | :x:                |
| D   | ResNet18 | AveragePool   | :x:                |
| E   | ResNet18 | NetVLAD       | :x:                |

## Citation
```
@InProceedings{Chen20,
  author       = "Honglie Chen and Weidi Xie and Andrea Vedaldi and Andrew Zisserman",
  title        = "VGGSound: A Large-scale Audio-Visual Dataset",
  booktitle    = "International Conference on Acoustics, Speech, and Signal Processing ",
  year         = "2020",
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
