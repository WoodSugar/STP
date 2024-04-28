cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/STFGNN/PeMS04/HyperSTFGNNnFC.py -g 0 &
python main/train.py -c model_setting/hyper_method/STFGNN/PeMS04/HyperSTFGNNnS.py -g 0 & 
python main/train.py -c model_setting/hyper_method/STFGNN/PeMS04/HyperSTFGNNnT.py -g 0 

