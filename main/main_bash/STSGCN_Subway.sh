cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STSGCN/Subway/STSGCN_Ori.py -g 0 & 
python main/train.py -c model_setting/compare_method/STSGCN/Subway/STSGCN_Ran.py -g 0 &
python main/train.py -c model_setting/compare_method/STSGCN/Subway/STSGCN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STSGCN/Subway/STSGCN_Uni.py -g 1 

