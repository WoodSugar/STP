cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/GWN/PeMS03/GWN_Ori.py -g 0 &
python main/train.py -c model_setting/compare_method/GWN/PeMS03/GWN_Idn.py -g 0 &

python main/train.py -c model_setting/compare_method/GWN/PeMS03/GWN_Uni.py -g 0 &
python main/train.py -c model_setting/compare_method/GWN/PeMS03/GWN_Ran.py -g 0