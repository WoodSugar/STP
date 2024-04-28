cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STFGNN/PeMS04/STFGNN_Ori.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/PeMS04/STFGNN_Ran.py -g 1 &
python main/train.py -c model_setting/compare_method/STFGNN/PeMS04/STFGNN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/PeMS04/STFGNN_Uni.py -g 1 

