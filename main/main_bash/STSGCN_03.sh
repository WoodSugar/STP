cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STSGCN/PeMS03/STSGCN_Ori.py -g 1 & 
python main/train.py -c model_setting/compare_method/STSGCN/PeMS03/STSGCN_Ran.py -g 1 &
python main/train.py -c model_setting/compare_method/STSGCN/PeMS03/STSGCN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STSGCN/PeMS03/STSGCN_Uni.py -g 1 

