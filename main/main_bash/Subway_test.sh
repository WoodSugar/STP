#!/bin/bash

cd /home/fangshen/ST_PreTrain/
source activate st_pre

python main/train.py -c model_setting/compare_method/LR_MySubway.py -g 0