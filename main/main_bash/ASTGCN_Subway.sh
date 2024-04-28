cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/ASTGCN/Subway/ASTGCN_Ori.py -g 0 &
python main/train.py -c model_setting/compare_method/ASTGCN/Subway/ASTGCN_Uni.py -g 1 