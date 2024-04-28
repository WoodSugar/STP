cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/DCRNN/PeMS03/DCRNN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/DCRNN/PeMS03/DCRNN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/DCRNN/PeMS03/DCRNN_Ran.py -g 1  