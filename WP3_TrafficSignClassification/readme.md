# GTSRB Training Pipeline 


## Table of Contents
1. [General Info](#1.-general-info)
2. [Technologies](#2.-technologies)
3. [Dataset](#3.-dataset)
4. [Models](#4.-models)
5. [Instructions](#5.-instructions)
8. [Contributors](#7.-Contributors)


## 1. General Info
This project contains the source code for training different image classification models on the 
[German-Traffic-Sign-Recognition-Benchmark (GTSRB)][official_dataset_site] dataset. Furthermore, it contains a component
for storing different prediction properties inside a pandas dataframe, which can be generated during the evaluation process. 

Note that the dataset is preprocessed, such that the ROI is cut out from the original image containing only the traffic 
sign (see [Annotation format][official_dataset_site]).

Source code used for experiments in the thesis ```Deep Learning for Vision-Based Perception in Automated Driving: Performance, Robustness,
and Monitoring``` by Yasin Bayzidi.


[official_dataset_site]: https://benchmark.ini.rub.de/gtsrb_dataset.html
[ki_absicherung]: https://www.ki-absicherung-projekt.de/


## 2. Technologies
Python version: 3.8.9

List of libraries used inside the code:   

* numpy             1.21.1
* pandas            1.3.1
* Pillow            8.4.0
* PyYAML            5.4.1
* setuptools        59.5.0
* tensorboard       2.6.0
* torch             1.10.1
* torchvision       0.11.2
* tqdm              4.62.0


## 3. Dataset
For more details about the GTSRB dataset, please visit the official [website][GTSRB website].

[GTSRB website]: https://benchmark.ini.rub.de/

The dataset can be downloaded [here][official_dataset].

[official_dataset]: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

Please download the following two zip files: ```GTSRB_Final_Training_Images.zip``` and ```GTSRB_Final_Test_Images.zip```.
Extract both folders into a new folder (e.g. ```data```).

Note that ```GTSRB_Final_Test_Images.zip``` does not include the image labels. These have to be downloaded from 
[kaggle][labels_download_link]. Please download the ```Test.csv``` file and place it inside the ```Image``` folder of the
```GTSRB_Final_Test_Images``` folder (the path should look like this: 
```./data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/Test.csv```).

[labels_download_link]: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Test.csv


## 4. Models
List of all models, which are supported for the GTSRB Training Pipeline:

* VGG16
* VGG19
* InceptionV3
* ResNet18
* ResNet50
* ResNet101
* ResNet152
* WideResNet50
* WideResNet101
* ResNeXt50
* ResNeXt101


## 5. Instructions
Make sure to download all the necessary libraries in the same version as shown in section [2. Technologies](#2-technologies)
to avoid conflicts and runtime errors.
Once the data has been downloaded and extracted like described in section [3. Dataset](#3-dataset), you can move on with 
the following steps:


### 5.1. Preprocessing the Data
Use ```generate_pickled_data.py``` to preprocess the dataset.
This script will crop the ROI and resize the images to a predefined size (we used 299x299 in order to have a consistent 
size, which is supported by all the models from [4. Models](#4-models)). It will also split the train dataset into a 
train and a validation dataset, depending on the specified proportion. Finally, it will generate a new folder 
(```pickled_data```), where it stores the train, validation and test dataset as pickle files (```train.p, valid.p, test.p```).  
When executing the script, you can specify the following arguments:
* ```--data_path``` 
  * this is the path to the folder, where you have extracted the GTSRB data (process described in section [3. Dataset](#3-dataset)).
* ```--size```  
  * this is the pixel size for resizing the images.
* ```--val_proportion```
  * this specifies the percentage of images from the train dataset, which should be used for creating the validation dataset.
* ```--seed```
  * the random seed for sampling the train images, which are used for the validation set.
* ```--destination_path```
  * this is the path, where to store the preprocessed train, validation and test dataset.   

Example execution: ```python3 generate_pickled_data.py --data_path=./data --size=299 --val_proportion=0.15 --seed=1 --destination_path=./data```

Outcome for the example from above:
```./data/pickled_data/train.p```, ```./data/pickled_data/valid.p```, ```./data/pickled_data/test.p```.


### 5.2. Training the model
Specify the train parameters inside the ```train_config.yaml```, which is located inside the config folder of this 
project. There you can specify which model should be trained, different hyperparameters, and you need to specify also
the path where your ```pickled_data``` folder is located as well as where to store the trained model weights. The 
training pipeline also stores the train progress and logs it to TensoBoard. Finally, you can execute the ```train.py``` 
script to run the training.   
When executing the script, you can specify the following argument:
* ```--config```
  * this is the path to the train_config.yaml file.
    
Example execution: ```python3 train.py --config=./config/train_config.yaml```

A summary of our training results is presented in section 
[6. Training Results](#6-training-results).

The training pipeline will create a train session folder which will be used for storing the model weights and 
TensorBoard logs. The folder name will contain the current datetime of the execution and the name of the model.   
Example: ```2021-08-12_20-14-47_ResNet18```   
Inside this folder will be a ```TensorBoard_Logs``` folder, a copy of the ```train_config.yaml``` file which was used 
for initiating the training and the final model weights whose file name contains the best accuracy that was achieved on 
the validation dataset during training (e.g. ```weights_100.0%.pt```).


### 5.3. Evaluating the model Performance
Once the training has finished, you can evaluate the model performance. Use the ```eval_config.yaml``` file, which is
located inside the config folder of this project. There you need to specify some evaluation parameters like the name
of the model which should be evaluated and the path to the model weights that were stored after the training. Please
refer to the comments inside the .yaml file, which describe each parameter in more detail. Once this is done, you can 
execute the script ```evaluate.py``` to start
the evaluation process. 
When executing the script, you can specify the following arguments:
* ```--config```
  * this is the path to the eval_config.yaml file.
* ```--extract_dataframe```
  * specifies whether the evaluation should store the prediction properties inside a pandas dataframe (Options: True or False).
* ```--target_layer```
  * specifies the name of the model layer (attribute from the model class), from which to extract the model activations.
  Only relevant if extract_dataframe is set to True.
* ```--mc_dropout_num_repeats```
  * amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only relevant if extract_dataframe is set
  to True.

Example execution: ```python3 evaluate.py --config=./config/eval_config.yaml --extract_dataframe=True 
--target_layer=avgpool --mc_dropout_num_repeats=20```

Once the evaluation is finished, it will print the prediction accuracy and loss for the specified dataset to the console 
and also log these results to TensorBoard.

If the argument ```extract_dataframe``` is set to True, the evaluation will create a pandas dataframe and store the 
following properties for each sample:

    'sample_id', 'label', 'img_clean', 'pred_clean', 'conf_clean', 'cscore_clean', 'layer_act_clean', 'cscore_mc_dropout_clean', 'cvariance_mc_dropout_clean'

- ```sample_id``` is used for identifying each sample (starting from 0 and counting upwards)
- ```label``` will contain the label for the current sample
- ```img_clean``` stores an RGB numpy array of the sample image
- ```pred_clean``` stores the class integer that was predicted by the model
- ```conf_clean``` contains the output softmax score (the highest score) for the predicted class as a float value
- ```cscore_clean``` contains all output softmax scores for all classes as float values inside a numpy array
- ```layer_act_clean``` contains the model activations from the target layer stored as float values inside a numpy array
- ```cscore_mc_dropout_clean``` contains the average softmax scores (prediction confidence) for each class based on all output scores from the Monte Carlo Dropout
- ```cvariance_mc_dropout_clean```  contains the variance of the prediction scores for each class based on all output scores from the Monte Carlo Dropout

This pandas dataframe will be stored as a .pickle file inside a new folder called ```evaluation_dataframes```, which will be 
located inside the train session folder besides the ```TensorBoard_Logs```. The file name will contain the name of the model,
the name of the dataset and the name of the target layer from which the activations where extracted 
(e.g. ```ResNet18_GTSRB_test_set_activations_layer_avgpool.p```).


### 5.4. TensorBoard
To inspect the train logs in TensorBoard, you need to open your Terminal and execute the TensorBoard command. This can 
be done by calling tensorboard and specifying a ```--logdir``` argument, which needs to point to a folder containing 
TensorBoard logs.

Example::

    tensorboard --logdir=/path/to/log_dir/model

The logdir path is in this case pointing to the train session folder of a single model that was trained.
If you want to compare multiple training session, you need to specify the parent folder as the logdir folder.

Example:

    tensorboard --logdir=/path/to/log_dir

Finally, you can view TensorBoard inside your browser by typing ```http://localhost:6006/``` in the address bar.

Further details regarding TensorBoard can be found [here][tensorboard].

[tensorboard]: https://pytorch.org/docs/stable/tensorboard.html


Parts of the training pipeline were taken from Barney Kim's GitHub repository
[traffic-sign-recognition][GitHub-Link].

[GitHub-Link]: https://github.com/wolfapple/traffic-sign-recognition