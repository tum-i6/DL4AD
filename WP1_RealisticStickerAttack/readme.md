# Realistic Sticker Attack 


## Table of Contents
1. [General Info](#1.-general-info)
2. [Technologies](#2.-technologies)
3. [Dataset](#3.-dataset)
4. [Models](#4.-models)
5. [Instructions](#5.-instructions)
7. [Contributors](#7.-Contributors)


## 1. General Info
This project contains the source code for attacking different image classification models (evaluating their robustness) 
using realistic real world stickers. The current version is focusing on attacking models trained on the 
[German-Traffic-Sign-Recognition-Benchmark (GTSRB)][official_dataset_site] dataset and is a follow-up project of the
[GTSRB Training Pipeline][GTSRB Training Pipeline]. Furthermore, it contains a component for storing different prediction 
properties inside a pandas dataframe, which can be generated during the attack evaluation process. 

Source code used for experiments in the thesis ```Deep Learning for Vision-Based Perception in Automated Driving: Performance, Robustness, 
and Monitoring``` Yasin Bayzidi.



[official_dataset_site]: https://benchmark.ini.rub.de/gtsrb_dataset.html
[ki_absicherung]: https://www.ki-absicherung-projekt.de/
[GTSRB Training Pipeline]: https://github.com/placeholder/gtsrb-training-pipeline


## 2. Technologies
Python version: 3.8.9

List of libraries used inside the code:    

* numpy             1.21.1
* opencv-python     4.5.3.56
* pandas            1.3.1
* Pillow            8.4.0
* PyYAML            5.4.1
* setuptools        59.5.0
* tensorboard       2.6.0
* torch             1.10.1
* torchvision       0.11.2
* tqdm              4.62.0


## 3. Dataset
This project comes with a small dataset divided in 3 folders:

* the ```./data``` folder contains 5 subfolders (```./data/1```,  ```./data/13```, ```./data/14```,  ```./data/17``` and 
```./data/28```), which contain train sample images of five GTSRB classes. Each of them contains 2 subfolders
(e.g. ```./data/1/train_set``` and ```./data/1/validation_set```), which hold exactly 40 train and 40 validation samples of 
the same traffic sign of size 299x299. These images represent the victim class, which is being attacked with 
the realistic stickers. When adding new images to these subfolders (or a new class) make sure to add exactly 40 images of 
size 299x299 containing the same traffic sign. It is also important to add images, that cover a wide variety of 
different angels, illuminations, distances, blurriness effects etc. This ensures a great robustness for the attack.

* the ```./masks``` folder contains the attack masks, used for constraining the stickers to be only applied on the traffic 
sign surface. There is a mask for every traffic sign shape. In case of using GTSRB, use the following masks:
  * ```./masks/baseline_mask_circle_shape.png``` for classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 
35, 36, 37, 38, 39, 40, 41, 42
  * ```./masks/baseline_mask_diamond_shape.png``` for classes: 12
  * ```./masks/baseline_mask_octagon_shape.png``` for classes: 14
  * ```./masks/baseline_mask_rotated_triangle_shape.png``` for classes: 13
  * ```./masks/baseline_mask_triangle_shape.png``` for classes: 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31

For the attack evaluation you need to specify the path to the GTSRB train, validation and test set. These are produced
by the [GTSRB Training Pipeline][GTSRB Training Pipeline] project. Make sure to generate the ```train.p```, 
```valid.p``` and ```test.p``` files before starting the attack evaluation process.

[GTSRB Training Pipeline]: https://github.com/placeholder/gtsrb-training-pipeline


## 4. Models
List of all models, which are supported for the Realistic Sticker Attack:

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

Note that an attack is being optimized for a specific model on a single traffic sign class by iteratively applying stickers
from the ```sticker_pool``` onto the train and validation images and evaluating the model's prediction performance.


### 5.1 Attack Types
There are two attack types:   
* ```target attacks```
  * These attacks have the goal of deceiving the model to predict a specific class. For the example case of attacking a 
  stop sign, the stickers produced by this attack can mislead the model to predict the target attack class 50km/h once
  applied to the stop sign. Note that this attack is very hard to optimize.
* ```misclassification attacks```
  * These attacks have the goal of deceiving the model but without targeting a specific class. The only goal is to get
  the prediction confidence for the true class as low as possible.


### 5.2 Attack Algorithms
The Realistic Sticker Attack comes with 3 different attack algorithms, which can be used for attacking the models:

* Saliency Sticker Application (SSA)
  * This attack algorithm produces an average saliency map for the model, based on the clean images from the 
  ```train_set``` folder. This saliency map is then used to place the stickers on the sign surface by covering all the 
  heatmap points. The algorithm also randomly rotates and resizes the stickers with respect to the constraints specified 
  in the config file. The algorithm iterates over all stickers from the ```sticker_pool``` and tries every texture out. This 
  process is repeated for many epochs to account for many sticker combinations and once this is done, it will store the 
  sticker setup that achieved the highest validation loss. This sticker set up will be then used to add another sticker 
  to it using the same process. The training is finished, once the algorithm has sampled the maximum amount of stickers 
  to be applied as specified in the config file.
* Monte Carlo Sticker Application (MCSA)
  * This attack algorithm works in the same way as the Saliency Sticker Application. The only difference is that instead of
  placing the stickers with respect to the saliency map in order to cover the heatmap points, this algorithm places the
  stickers completely randomly somewhere on the sign surface. This approach gives the algorithm even more space for 
  finding the weak points.
* Random Sticker Application (RSA)
  * This attack algorithm is the fastest one, since it does not optimize any loss at all and just places the stickers 
  randomly on the sign surface. It also does not iterate over all stickers from the ```sticker_pool```, but rather samples
  random sticker textures that are directly applied to the traffic sign. 

Note that in case of a target attack, the SSA and MCSA algorithms minimze the loss for predicting the target attack 
class instead of maximizing the loss for the label class.


### 5.3 Optimizing (Training) the Attack
Specify the attack parameters inside the ```realistic_sticker_attack_config.yaml```, which is located inside the config 
folder of this project. There you can specify the victim model, which attack algorithm to use, which class to attack, 
whether the attack should be a target attack or just mislassification of the victim class etc.
The attack also stores the optimization progress and logs it to TensoBoard. Finally, you can execute the 
```realistic_sticker_attack.py``` script to run the training.   
When executing the script, you can specify the following argument:
* ```--config```
  * this is the path to the realistic_sticker_attack_config.yaml file.

Example execution: ```python3 realistic_sticker_attack.py --config=./config/realistic_sticker_attack_config.yaml```

The attack will create a train session folder which will be used for storing the final sticker overlays and 
TensorBoard logs. The folder name will contain the current datetime of the execution, the name of the victim model,
the name of the attack algorithm and the victim class (and also the attack target class if specified).  
Example: ```2021-12-23_09-21-30_ResNet18_MCSA_on_class_14```   
Inside this folder will be a ```TensorBoard_Logs``` folder, a copy of the ```realistic_sticker_attack_config.yaml``` file 
which was used for initiating the attack and a folder called ```sticker_overlays```. Inside this folder you will find the 
sticker overlay files which have been generated by the attack starting from one sticker up to the maximum amount of stickers 
specified by the config file. These overlays will be transparent png files of size 299x299, that can be applied over
the traffic sign image. The name of the png files will contain the name of the victim model, the name of the attack
algorithm, the amount of stickers present in the sticker overlay, the prediction performances on the (small scale) train
and validation dataset (for the label class) when this sticker overlay is applied to them, and finally the sticker 
coverage with respect to the traffic sign surface area.    
Example: ```ResNet18_MCSA_5_sticker_attack_true_target_train_100.0%_val_97.5%_coverage_16.6%.png```


### 5.4 Evaluating the Attack
Once the attack optimization has finished, you can evaluate the model performance using the sticker overlays. 
Use the ```evaluate_attack_config.yaml``` file, which is located inside the config folder of this project. There you 
need to specify some evaluation parameters like the name of the model which should be evaluated and the path to the 
attack session folder, which was produced by the attack optimization process. Furthermore, you need to specify also a 
csv_path which will be used for storing a csv files holding the evaluation results. Please refer to the comments inside 
the .yaml file, which describe each parameter in more detail. Once this is done, you can execute the script 
```evaluate_attack.py``` to start the evaluation process.   
When executing the script, you can specify the following arguments:
* ```--config```
  * this is the path to the evaluate_attack_config.yaml file.
* ```--extract_dataframe```
  * specifies whether the evaluation should store the prediction properties inside a pandas dataframe (Options: True or False).
* ```--target_layer```
  * specifies the name of the model layer (attribute from the model class), from which to extract the model activations.
  Only relevant if extract_dataframe is set to True.
* ```--mc_dropout_num_repeats```
  * amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only relevant if extract_dataframe is set
  to True.
* ```--file_name```
  * Specifies the name of a single sticker overlay file, which should be evaluated. If not specified the evaluation will 
  evaluate every sticker overlay file inside the sticker_overlays folder.

Example execution: ```python3 evaluate_attack.py --config=./config/evaluate_attack_config.yaml --extract_dataframe=True 
--target_layer=avgpool --mc_dropout_num_repeats=20  --file_name=ResNet18_RSA_5_sticker_attack_true_target_train_100.0%_val_100.0%_coverage_10.0%.png```

Note that during the evaluation, only the samples from the victim class will be used from the specified dataset.

Once the evaluation is finished, it will print the prediction accuracy and loss for the attacked dataset to the console.
The final evaluation results will be also stored inside a csv file, which is located depending on your csv_path argument
inside the config file. The csv file will contain the following information (columns):

    'attack session path', 'sticker overlay file name', 'victim model name', 'victim class', 'attack target class', 'attack algorithm name', 'amount of sticker', 'sticker coverage', 'evaluation model name', 'dataset type', 'amount of evaluation samples', 'accuracy on clean samples', 'loss on clean samples', 'accuracy on attacked samples', 'loss on attacked samples', 'accuracy for target attack', 'loss for target attack', amount of class 0 predictions, amount of class 1 predictions, ..., amount of class 42 predictions

- ```attack session path``` the path to the attack session folder as specified in the config
- ```sticker overlay file name``` name of the png file from the sticker_overlays folder, which has been evaluated
- ```victim model name``` name of the model on which the attack has been optimized
- ```victim class``` victim class which has been attacked 
- ```attack target class``` the class which should be predicted in case of a target attack (set to False if only mislassification is to be achieved)
- ```attack algorithm name``` the name of the attack algorithm (MCSA, SSA or RSA)
- ```amount of sticker``` the amount of stickers inside the png sticker overlay file
- ```sticker coverage``` the sticker coverage with respect to the area of the victim traffic sign
- ```evaluation model name```  the name of the model which was used for evaluating the attack
- ```dataset type``` the type of the dataset used for te evaluation (train, valid or test)
- ```amount of evaluation samples``` the amount of samples inside the dataset, which belong to the victim class
- ```accuracy on clean samples``` prediction accuracy on the clean victim class samples from the dataset
- ```loss on clean samples``` prediction loss on the clean victim class samples from the dataset
- ```accuracy on attacked samples``` prediction accuracy on the attacked victim class samples from the dataset
- ```loss on attacked samples``` prediction loss on the attacked victim class samples from the dataset
- ```accuracy for target attack```  prediction accuracy on the attacked victim class samples from the dataset with respect to the attack target class label
- ```loss for target attack```  prediction loss on the attacked victim class samples from the dataset with respect to the attack target class label
- ```amount of class 0 predictions``` tracks the amount of predictions of class 0
- ```amount of class 1 predictions``` tracks the amount of predictions of class 1
- ...
- ```amount of class 42 prediction``` tracks the amount of prediction of class 42

If the argument ```extract_dataframe``` is set to True, the evaluation will create a pandas dataframe and store the 
following properties for each sample:

    'sample_id', 'label', 'img_clean', 'pred_clean', 'conf_clean', 'cscore_clean', 'layer_act_clean', 'cscore_mc_dropout_clean', 'cvariance_mc_dropout_clean'

- ```sample_id``` is used for identifying each sample (starting from 0 and counting upwards)
- ```label``` will contain the label for the current sample
- ```img_attacked``` stores an RGB numpy array of the sample image
- ```pred_attacked``` stores the class integer that was predicted by the model
- ```conf_attacked``` contains the output softmax score (the highest score) for the predicted class as a float value
- ```cscore_attacked``` contains all output softmax scores for all classes as float values inside a numpy array
- ```layer_act_attacked``` contains the model activations from the target layer stored as float values inside a numpy array
- ```cscore_mc_dropout_attacked``` contains the average softmax scores (prediction confidence) for each class based on all output scores from the Monte Carlo Dropout
- ```cvariance_mc_dropout_attacked```  contains the variance of the prediction scores for each class based on all output scores from the Monte Carlo Dropout

This pandas dataframe will be stored as a .pickle file inside a new folder called ```evaluation_dataframes```, which will be 
located inside the attack session folder besides the ```TensorBoard_Logs``` and ```sticker_overlays```.
The file name will contain the name of the model, the name of the dataset, the name of the target layer from which the
activations where extracted, the victim class and the amount of stickers used for the attack
(e.g. ```ResNet18_GTSRB_test_set_activations_layer_layer4_victim_class_14_sticker_amount_1.p```).


### 5.5 TensorBoard
To inspect the attack logs in TensorBoard, you need to open your Terminal and execute the TensorBoard command. This can 
be done by calling tensorboard and specifying a ```--logdir``` argument, which needs to point to a folder containing 
TensorBoard logs.

Example::

    tensorboard --logdir=/path/to/experiment_output/attack1

The logdir path is in this case pointing to a single attack session folder.
If you want to compare multiple attack session, you need to specify the parent folder as the logdir folder.

Example:

    tensorboard --logdir=/path/to/realistic_sticker_attack/experiment_outputs

Finally, you can view TensorBoard inside your browser by typing ```http://localhost:6006/``` in the address bar.

Further details regarding TensorBoard can be found [here][tensorboard].

[tensorboard]: https://pytorch.org/docs/stable/tensorboard.html
