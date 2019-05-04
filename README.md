# Receptive Field Blocks Text Detection Module (RFBTD Module)
A Dense Text Detection model using Receptive Field Blocks

### Introduction
This repo contains text detection based on [Receptive Field Blocks](https://arxiv.org/abs/1711.07767). The text detection model provides a dense receptive field, for predicting text boxes in dense natural scene images like documents, articles etc.

The model is also inspired from [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2), where the **RRBOX** part and loss function is taken from.

The features of the model are summarized below:
+ Keras implementation for lucid and clean code.
+ Backbone: **Resnet50**
+ Inference time for 720p images:

    **GPU VERSION**
    + Graphics Card: MX130
    + Inference Time: 700ms
    + Batch Size: 1
   
    **CPU VERSION**
    + CPU: Intel i7-8550U CPU @ 1.80GHz
    + Inference Time: 1750ms
    + Batch Size: 1

+ The pre-trained model provided achieves **47.09**(Single Crop, Resize Only) F1-score on ICDAR 2015, but was not trained on ICDAR 2015. To improve accuracy fine-tune on ICDAR 2015, and predict with multiple crops.

+ The model is tuned for predicting text boxes for natural scene documents, like bank statements, forms, recipts, etc, and evidently do OCR on these text boxes.


Please cite these [paper](https://arxiv.org/abs/1711.07767), [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Demo](#demo)
3. [Eval](#eval)
4. [Examples](#examples)

### Installation
+ Requirements from requirements.txt
     tensorflow==1.13.1
     Keras==2.2.4
     numpy==1.16.2
     opencv_contrib_python==4.0.0.21
     plumbum==1.6.7

### Download
Pre-trained Model Link [GoogleDrive](https://drive.google.com/open?id=1mw8v_VV1KidyrqY_0A_oSYxRhPex4oKY)

### Demo
If you've downloaded the pre-trained model, run 
```
python3 run_demo.py
```
the images are taken from **test_images/input_images** and the output is predicted to **test_images/predicted_images**

### Eval
If you want to benchmark it on ICDAR 2015, run 
```
python eval.py
```
the images are taken from **eval_images/evaluation_images** and the output is predicted to **eval_images/predicted_boxes**, text files will be then written to the output path. This format confides to the ICDAR text detection challenge format.

### Examples
Here are some examples,
![image_1](test_images/predicted_images/1.jpg)
![image_2](test_images/predicted_images/2.jpg)
![image_3](test_images/predicted_images/3.jpg)
![image_4](test_images/predicted_images/4.jpg)

### Issues
If you encounter any issues, please create an issue tracker.
