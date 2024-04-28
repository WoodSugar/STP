cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/DCRNN/METR/DCRNN_OriNoGc.py -g 0 &
python main/train.py -c model_setting/compare_method/DCRNN/METR/DCRNN_IdnNoGc.py -g 0 &
python main/train.py -c model_setting/compare_method/DCRNN/METR/DCRNN_UniNoGc.py -g 0 &
python main/train.py -c model_setting/compare_method/DCRNN/METR/DCRNN_RanNoGc.py -g 0  