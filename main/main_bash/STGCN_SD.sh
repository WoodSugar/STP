cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STGCN/SD/STGCN_Ori.py -g 0 & 
python main/train.py -c model_setting/compare_method/STGCN/SD/STGCN_Ran.py -g 0 &
python main/train.py -c model_setting/compare_method/STGCN/SD/STGCN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STGCN/SD/STGCN_Uni.py -g 1 

