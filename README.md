# U-Net-TensorFlow
This repository is a TensorFlow implementation of the ["U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI2015](https://arxiv.org/pdf/1505.04597.pdf). It completely follows the original U-Net paper.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/58258734-3cd6c380-7dae-11e9-8b13-7bed307b3981.png" width=700)
</p>  

## EM Segmentation Challenge Dataset 
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59033695-3e22e880-88a4-11e9-8034-5abc27146a76.png" width=1000)
</p>  

## Requirements
- tensorflow 1.13.1
- python 3.5.3  
- numpy 1.15.2  
- scipy 1.1.0
- tifffile 2019.3.18
- opencv 3.3.1
- matplotlib 2.2.2
- elasticdeform 0.4.4
- scikit-learn 0.20.0

## Implementation Details
This implementation completely follows the original U-Net paper from the following aspects:
- Input image size 572 x 572 x 1 vs output labeled image 388 x 388 x 2
- Upsampling used fractional strided convolusion (deconv)
- Reflection mirror padding is used for the input image
- Data augmentation: random translation, random horizontal and vertical flip, random rotation, and random elastic deformation
- Loss function includes weighted cross-entropy loss and regularization term
- Weight map is calculated using equation 2 of the original paper
- In test stage, this implementation achieves average of the 7 rotated versions of the input data

## Examples of the Data Augmentation
- Random Translation
- Random Horizontal and Vertical Flip
- Random Rotation
- Random Elastic Deformation

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59035260-db335080-88a7-11e9-8434-0714e611418c.png" width=1000)
</p>  
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59035357-0e75df80-88a8-11e9-96b1-8c5a6f39179a.png" width=1000)
</p>  
  
## Fundamental of the Different Sized Input and Output Images in Training Process
- Reflected mirror padding is utilized first (white lines indicate boundaries of the image)
- Randomly cropping the input image, label image, and weighted image
- Blue rectangle region of the input image and red rectangle of the weight map are the inputs of the U-Net in the training, and the red rectangle of the labeled image is the ground-truth of the network.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59035950-4fbabf00-88a9-11e9-8976-2083f6290746.png" width=1000)
</p>  

## Test Paradigm
- In test stage, each test image is the average of the 7 rotated version of the input data. The final prediction is the averaging the 7 predicted restuls.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59038491-15075580-88ae-11e9-9f49-08b3ba04c358.png" width=1000)
</p>  
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59038767-8c3ce980-88ae-11e9-801d-6ba002747a0b.png" width=1000)
</p>    

- For each rotated image, the four regions are extracted, top left, top right, bottom left, and bottom right of the each image to go through the U-Net, and the prediction is calculated averaging the overlapping scores of the four results

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59037210-c9ec4300-88ab-11e9-95a6-a12dd30134fb.png" width=600)
</p>  
  
**Note**: White lines indicate boundaries of the image. 

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59037900-1f751f80-88ad-11e9-9c16-7da62d763a7f.png" width=600)
</p>  

**Note:** The prediciton results of the EM Segmentation Challenge Test Dataset  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59039864-952eba80-88b0-11e9-8c49-178cb87bd250.png" width=1000)
</p>  

## Download Dataset
Download the EM Segmetnation Challenge dataset from [ISBI challenge homepage](http://brainiac2.mit.edu/isbi_challenge/).

## Documentation
### Directory Hierarchy
``` 
.
│   U-Net
│   ├── src
│   │   ├── dataset.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Data
│   └── EMSegmentation
│   │   ├── test-volume.tif
│   │   ├── train-labels.tif
│   │   ├── train-wmaps.npy (generated in preprocessing)
│   │   └── train-volume.tif
```  
### Preprocessing
Weight map need to be calculated using segmentaion labels in training data first. Calculaing wegith map using on-line method in training will slow down processing time. Therefore, calculating and saving weighted map first, the weight maps are augmented according to the input and label images. Use `preprocessing.py` to calculate weight maps. Example usage:
```
python preprocessing.py
```

### Training U-Net
Use `main.py` to train the U
```
python main.py
```
- `gpu_index`: gpu index if you have multiple gpus, default: `0`  
- `dataset`: dataset name, default: `EMSegmentation`
- `batch_size`: batch size for one iteration, default: `4`
- `is_train`: training or inference (test) mode, default: `True (training mode)`  
- `learning_rate`: initial learning rate for optimizer, default: `1e-3` 
- `weight_decay`: weight decay for model to handle overfitting, default: `1e-4`
- `iters`: number of iterations, default: `20,000`  
- `print_freq`: print frequency for loss information, default: `10`  
- `sample_freq`: sample frequence for checking qualitative evaluation, default: `100`
- `eval_freq`: evaluation frequency for evluation of the batch accuracy, default: `200`
- `load_model`: folder of saved model that you wish to continue training, (e.g. 20190524-1606), default: `None`  

### Test U-Net
Use `main.py` to test the models. Example usage:
```
python main.py --is_train=False --load_model=folder/you/wish/to/test/e.g./20190524-1606
```  
Please refer to the above arguments.

### Tensorboard Visualization
**Note:** The following figure shows data loss, weighted data loss, regularization term, and total loss during training process. The batch accuracy also is given in tensorboard.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/59041940-8c3fe800-88b4-11e9-9d8a-a4c5345b6718.png" width=1000)
</p>  

### Citation
```
  @misc{chengbinjin2019u-nettensorflow,
    author = {Cheng-Bin Jin},
    title = {U-Net Tensorflow},
    year = {2019},
    howpublished = {\url{https://github.com/ChengBinJin/U-Net-TensorFlow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [Zhixuhao](https://github.com/zhixuhao/unet)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)
  
 ## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
