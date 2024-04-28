cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/DCRNN/SD/DCRNN_OriNoGc.py -g 1 &
python main/train.py -c model_setting/compare_method/DCRNN/SD/DCRNN_IdnNoGc.py -g 1 &
# python main/train.py -c model_setting/compare_method/DCRNN/SD/DCRNN_UniNoGc.py -g 1 &
# python main/train.py -c model_setting/compare_method/DCRNN/SD/DCRNN_RanNoGc.py -g 1  