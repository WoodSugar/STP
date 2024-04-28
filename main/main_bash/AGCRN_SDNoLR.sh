cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_OriNoLR.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_UniNoLR.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_IdnNoLR.py -g 0 &
python main/train.py -c model_setting/compare_method/AGCRN/SD/AGCRN_RanNoLR.py -g 0