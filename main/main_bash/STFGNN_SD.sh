cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STFGNN/SD/STFGNN_Ori.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/SD/STFGNN_Ran.py -g 1 &
python main/train.py -c model_setting/compare_method/STFGNN/SD/STFGNN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STFGNN/SD/STFGNN_Uni.py -g 1 

