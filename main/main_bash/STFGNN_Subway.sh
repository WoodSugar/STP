cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STFGNN/Subway/STFGNN_Ori.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/Subway/STFGNN_Ran.py -g 1 &
python main/train.py -c model_setting/compare_method/STFGNN/Subway/STFGNN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/Subway/STFGNN_Uni.py -g 1 

