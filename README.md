# STP
Source Code the proposed Spatio-Temporal Propagation (STP) module.

This work is to be submitted in a journal or a conference, and the all of the source codes are being updated.

## We have provided the following information for better reproducing
1. Pre-processed dataset which is publicly available on Google Drive [here](https://drive.google.com/drive/folders/1P05v64MnhEM1arhJA-DjpmeHainqrvaN?usp=drive_link)
    - Put the downloaded data into the [`dataset`](./dataset/) folder.
    - Query the description file in [`dataset_describe`](./dataset_describe/) folder to know the data format.
2. All the training settings for reproducing the results
    - Model structure is in [`compare_model`](./compare_model/) folder.
    - Model settings are in [`model_setting`](./model_setting/) folder, where baselines are in the [`compare_method`](./model_setting/compare_method/) folder and the STP enhanced models are in the [`hyper_method`](./model_setting/hyper_method/) folder.


3. Training logs, trained models, and evaluation results of partial methods are stored on Google Drive [here](https://drive.google.com/drive/folders/1mmesNYG_iaQ3LNavQejEEhpgXc780dZM?usp=drive_link) (about 1.5GB).

    - Please download and unzip the `result.zip` file and put them in the [`result`](./result/) folder. 
    - The following files are avalible on each experiment folder:
        - tensorboard logs (`tensorboard/` folder)
        - identity config file (`cfg.txt`)
        - last epoch model (`*_[MAX_EPOCH].pt`)
        - best model on valid dataset (`*_best_val_MAE.pt`)
        - saved setting file to reproduce (`*.py`)
        - training log (`training_log_*.log`)
        - evaluation results(`validate_result_*.log`)

4. Training and testing scripts have two options:
    - **Recommend.** Run the training script with the command line as follows. This contains three functions: 
        - Train the model from scratch. 
        - Resume training if not finished with the last epoch. 
        - Test the best model from the checkpoint if exists.

        ```python
        python main/train.py -c [YOUR MODEL SETTING]
        ```
        
    - **Not Recommend.** Run the testing script with the command line as follows. This requires to specify the checkpoint file.
    
        ```python
        python main/test.py -c [YOUR MODEL SETTING] -ck [YOUR MODEL CHECKPOINT]
        ```
        - You can replace the option `[YOUR MODEL CHECKPOINT]` with the corresponding checkpoint file path such as `FILE_PATH/*_best_val_MAE.pt`.



## We have directly released the following results (Stored on Google Drive)

Total result files are more than 60GB and cannot be uploaded to GitHub or any other online storage. 

Thus only partial results are provided on Google Drive [here](https://drive.google.com/drive/folders/1mmesNYG_iaQ3LNavQejEEhpgXc780dZM?usp=drive_link) (about 1.5GB) as follows.

### Re-investigation on current methods
* Results of feature normalization on LR as baseline. 
    
    Path: [`result/compare_model/LR`](./result/compare_model/LR)

* Results of feature normalization on MLP as baseline. 

    Path: [`result/compare_model/MLP`](./result/compare_model/MLP)

* Results of replacing the adjacency matrix on STSGCN as baseline. 

    Path: [`result/compare_model/STSGCN`](./result/compare_model/STSGCN)

* Results of replacing the adjacency matrix on STFGNN as baseline.
    
    Path: [`result/compare_model/STFGNN`](./result/compare_model/STFGNN)

### Efficacy of STP as both plug-in tool and substitution of GCNs on advanced methods

* Results of STP as plug-in tool and ablation studies on LR.

    Path: [`result/hyper_model/LR`](./result/hyper_model/LR)

* Results of STP as plug-in tool and ablation studies on MLP.

    Path: [`result/hyper_model/MLP`](./result/hyper_model/MLP)

* Results of STP as plug-in tool on STSGCN.

    Path: [`result/hyper_model/STSGCN`](./result/hyper_model/STSGCN)

* Results of STP as substitution of GCNs on STSGCN.

    Path: [`result/hyper_model/HyperSTSGCN`](./result/hyper_model/HyperSTSGCN)

* Results of STP as plug-in tool and ablation studies on STFGNN.

    Path: [`result/hyper_model/STFGNN`](./result/hyper_model/STFGNN)

* Results of STP as substitution of GCNs and ablation studies on STFGNN.

    Path: [`result/hyper_model/HyperSTFGNN`](./result/hyper_model/HyperSTFGNN)


## More details are on the way
1. Necessary environment for running the code 
2. We are working on the code to make it more readable and efficient.

## Acknowledgement
We appreciate the [EasyTorch](https://github.com/cnstark/easytorch) and [BasicTS](https://github.com/zezhishao/BasicTS) toolboxes to support this work.