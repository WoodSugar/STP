cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/STFGNN/SD/HyperSTFGNNnS.py -g 0 & 
python main/train.py -c model_setting/hyper_method/STFGNN/SD/HyperSTFGNNnT.py -g 0 & 
python main/train.py -c model_setting/hyper_method/STFGNN/SD/HyperSTFGNNnFC.py -g 0
