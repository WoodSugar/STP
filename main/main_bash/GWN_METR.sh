cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/GWN/METR/GWN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/METR/GWN_Idn.py -g 1 &

python main/train.py -c model_setting/compare_method/GWN/METR/GWN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/METR/GWN_Ran.py -g 1