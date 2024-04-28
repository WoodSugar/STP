cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/GWN/Subway/GWN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/Subway/GWN_Idn.py -g 1 &

python main/train.py -c model_setting/compare_method/GWN/Subway/GWN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/Subway/GWN_Ran.py -g 1