cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/STGCN/PeMS04/STGCN_Ori.py -g 0 & 
python main/train.py -c model_setting/compare_method/STGCN/PeMS04/STGCN_Ran.py -g 0 &
python main/train.py -c model_setting/compare_method/STGCN/PeMS04/STGCN_Idn.py -g 1 & 
python main/train.py -c model_setting/compare_method/STGCN/PeMS04/STGCN_Uni.py -g 1 

