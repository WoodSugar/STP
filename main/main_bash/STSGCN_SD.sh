cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STSGCN/SD/STSGCN_Ori.py -g 0 & 
python main/train.py -c model_setting/compare_method/STSGCN/SD/STSGCN_Ran.py -g 0 &
python main/train.py -c model_setting/compare_method/STSGCN/SD/STSGCN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STSGCN/SD/STSGCN_Uni.py -g 1 

