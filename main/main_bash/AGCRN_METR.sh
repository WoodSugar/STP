cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/AGCRN/METR/AGCRN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/METR/AGCRN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/METR/AGCRN_Idn.py -g 1 &
python main/train.py -c model_setting/compare_method/AGCRN/METR/AGCRN_Ran.py -g 1