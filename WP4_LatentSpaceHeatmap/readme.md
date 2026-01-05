

### Abstract
Source code used for experiments in the thesis ```Deep Learning for Vision-Based Perception in Automated Driving: Performance, Robustness,
and Monitoring``` by Yasin Bayzidi.


This project investigates the effect of adding attention loss to the latent representations of the deep learning-based detectors. To do so, two models will be trained per detector architecture per dataset: the baseline model and the one trained with the new loss term. After the training is done, the models will be evaluted from different perspectives to compare their performance differences.

This repository supports the following datasets and detectors:


|                  | Faster RCNN | Mask RCNN | FCOS | RetinaNet | SSD:tw-26a0: |
| :--------------: | :---------: | :-------: | :--: | :-------: | :-: |
|   CityPersons   |      x      |     x     |  x  |     x     |  x  |
| EuroCity Persons |      x      |    N/A    |  x  |     x     |  x  |
|       KIA       |      x      |     x     |  x  |     x     |  x  |

:tw-26a0: Note that the SSD is different from the other detectors in terms of the backbone design, here the SSD models are implemented to only keep the consistance with the previous PLF paper and they have inferior performance.



For different datasets, the mappings between the actual subsets used in this project and the official subsets are slightly different:

|                  | actually used training set | actually used val set  | actually used val set  | 
| :--------------: | :---------: | :-------: | :--: | 
|   CityPersons   |      official CP training set minus 300 images      |     300 images from the official CP training set     |  official CP val set |   
| EuroCity Persons |       official ECP training set      |     official ECP val set    |   official ECP val set  |    
|       KIA       |      official KIA training set      |    official KIA val set     |  official KIA test set  |   

### Training

To train the models, only the training config file and the training script `train.py` are needed.

#### Training configs

The training configs include the necessary parameters setting for the training. For different datasets, their corresponding training configs are seperately made as the following:


|                  |             Config             |
| :--------------: | :----------------------------: |
|   CityPersons   | `config/train_config_cp.yaml` |
| EuroCity Persons | `config/train_config_ecp.yaml` |
|       KIA       | `config/train_config_kia.yaml` |

Inside of the config files, there are various parameter settings. Please look into the training config files to know more about the parameters usages. 

In these parameters, the `contrastive_loss_weighting_factor` is used to balance the new loss with the standard losses of the baseline detectors. It's recommand to first use a small weighting factors such as 0.05 or 0.1, and a grid search for determining the optimal weighting factors for each detector should be made if applicable. 

#### Start training
After setting the desired parameters in the config files, the training can be easily done by calling the `train.py` script. The script supports either training from the first epoch or resuming the training from a breakpoint epoch. This can be controlled by passing the correspoding input arguments to the `train.py`.

##### Training from the 1st epoch
In this case only the config files is needed. Training can be done by doing the following commands in the terminal:

`python3 train.py --config=.config/train_config_cp.yaml`

After runing the above command, the folder including the trained models will be created under the path specified by the `output_path` parameter in the config file. The format of the folder is consisted of the following parts:

`created_date_dataset_name_model_name_task_weighting_factor`

Here are two examples:

`2023-02-07_12-54-52_CityPersons_FasterRCNN-ResNet50-OD_weight_0`

`2023-03-04_16-59-19_CityPersons_MaskRCNN-ResNet50-IS_weight_10`

Note that the weighting factors are formatted in percentage in the namings: 0 means the baseline model, and 10 means the weighting factor namely the  `contrastive_loss_weighting_factor` in the config file is 0.1. Regarding the performed tasks, the abbreviations are listed as the following:

- object detection (**OD**)
- keypoint detection (**KD**)
- instance segmentation (**IS**)

Note that in this project only the object detection results are evaluated even the model supports other tasks besides OD, e.g., Mask RCNN for instance segmentation. 

##### Resume training from a breakpoint epoch
Sometime the training process is paused, and only a few epochs are trained. In this case a resumed training may help. For resuming the training, only the path of the folder name is needed, and this folder should include a `train_config.yaml` file as the config file which should be created during the precious unfinished training process. 

To resume the training from a specific folder, run the following command:

`python3 train.py --resume=/path/to/output_dir`

The script will infer the lastest finished epoch and adjust the learning rate etc. to finish the training at the desired epoch number specified in the `train_config.yaml` file. 


##### Content of the folder
After finishing the training process, the folder will have the following things:

1. the saved checkpoints from 0 to the desired ending epoch
2. `final_model_coco.pth`: the model weight achieving the highest COCO AP at the val-set during the training phase. **Recommand to use it as the final trained model. **
3. `final_model_f1.pth`:  the model weight achieving the highest F1 score at the val-set during the training phase. 
4. `TensorBoard_Logs` folder: include the logs containing the losses during the training and the evaluation results on the val-set for each epoch. 

For using the tensorbard, run the following command in the server's terminal:

`tensorboard --logdir=/path/to/output_dir/TensorBoard_Logs`
It will output link to see the tensorboard, but it cannot be opened yet cuz it's till on the server.

And then open a new local terminal to run the following command:

ssh -N -f -L localhost:6006:localhost:6006 user@remote-server


Then open the link in local brower to see the tensorboard. 

Untill this step, the training phase is done, the next part is evaluation. 


### Evaluation

For evaluation, there is a config file `config/eval_config.yaml` needed to be specified, and it includes the path of the trained model, the dataset path etc. Please look into this config file to know more about the usage of the parameters.

The evaluation process is done by calling the `evaluation.py` script. This script supports the following features:

1. evaluate the models in terms of APs and LAMR etc. and save the PR curves etc.
2. save the visualization images to the subfolder.
3. extract the PLFs dataframe

All these features are intergrated into the `evaluation.py` script and they can be controlled by the input arguments. The input arguments are listed as below:

- **config**: the path of the `config/eval_config.yaml`
- **subset**: evaluated on which subset. Options include train, val and test. Default is test
- **no-eval**: if evaluate the model on the subset. Only evaluate when this argument is not flagged, default evaluate. 
- **save-vis**: if save visualization for the first 10 images in the subset. Only save them when this argument is flagged, default not save.
- **plfs**: if extract the PLFs values for the images in the subset and save them as a dataframe. Only save them when this argument is flagged, default not save.

Examples:

`python3 evaluation.py --config=.config/eval_config.yaml --subset=test --save-vis --plfs`
Above command will first evalute the model on the testset, and then save a couple of visualization images, and finally extract the PLFs values and save it as a dataframe. 

`python3 evaluation.py --config=.config/eval_config.yaml --no-eval --plfs`
Above command will only extract the PLFs values for the testset and save it as a dataframe. 

``python3 evaluation.py --config=.config/eval_config.yaml --no-eval``
Above command will do nothing. 

Due to `config/eval_config.yaml` will only contain one model and one dataset at one time, so an unique tag can be infered from the given `config/eval_config.yaml`file, and this tag can help naming the various reults. 

The format of the tag is : `tag = dataset_name_model_name_wf_contra_weighting_factor'`

The weighting factor is in percentage, below are couple of exmaple tags:
- `CityPersons_FCOS_wf_10`: FCOS model trained on the CityPersons dataset with a weighting factor os 0.1

- `EuroCityPersons_MaskRCNN_wf_0`: Mask RCNN model trained on the EuroCityPersons dataset with a weighting factor of 0 (baseline model)
