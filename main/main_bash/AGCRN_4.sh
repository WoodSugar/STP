cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/AGCRN/PeMS04/AGCRN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/PeMS04/AGCRN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/PeMS04/AGCRN_Idn.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/PeMS04/AGCRN_Ran.py -g 1