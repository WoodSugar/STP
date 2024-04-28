cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/GWN/SD/GWN_OriNoAda.py -g 0 &
python main/train.py -c model_setting/compare_method/GWN/SD/GWN_IdnNoAda.py -g 0 &

python main/train.py -c model_setting/compare_method/GWN/SD/GWN_UniNoAda.py -g 0 &
python main/train.py -c model_setting/compare_method/GWN/SD/GWN_RanNoAda.py -g 0