cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STSGCN/METR/STSGCN_Ori.py -g 0 & 
python main/train.py -c model_setting/compare_method/STSGCN/METR/STSGCN_Ran.py -g 0 &
python main/train.py -c model_setting/compare_method/STSGCN/METR/STSGCN_Idn.py -g 0 & 
python main/train.py -c model_setting/compare_method/STSGCN/METR/STSGCN_Uni.py -g 0 

