cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_Idn.py -g 1 &

python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_Ran.py -g 1