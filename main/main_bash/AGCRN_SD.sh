cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_Ori.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_Uni.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_Idn.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_Ran.py -g 0