# STP
Source Code the proposed Spatio-Temporal Propagation (STP) module.

This work is to be submitted in a journal or a conference, and the all of the source codes are being updated.

## We have provided the following information for better reproducing
1. Preprocessed dataset which is publicly available on Google Drive [here](https://drive.google.com/drive/folders/1P05v64MnhEM1arhJA-DjpmeHainqrvaN?usp=drive_link)
    - Put the downloaded data into the [`dataset`](./dataset/) folder.
    - Query the description file in [`dataset_describe`](./dataset_describe/) folder to know the data format.
2. All the training settings for reproducing the results
    - Model structure is in [`compare_model`](./compare_model/) folder.
    - Model settings are in [`model_setting`](./model_setting/) folder, where baselines are in the [`compare_method`](./model_setting/compare_method/) folder and the STP enhanced models are in the [`hyper_method`](./model_setting/hyper_method/) folder.

3. Training and testing scripts has two options:
    - run the 

## We are going to provide the following information for better reproducing
1. Necessary packages for running the code 
2. Model settings for reproducing the results
3. Training logs and scripts
4. Trained models for reproducing the results
5. Tesing scripts and results


Acknowledgement
We appreciate the [EasyTorch](https://github.com/cnstark/easytorch) and [BasicTS](https://github.com/zezhishao/BasicTS) toolboxes to support this work.