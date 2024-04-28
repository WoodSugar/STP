cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/DCRNN/PeMS04/DCRNN_IdnNoGc.py -g 0 &
python main/train.py -c model_setting/compare_method/DCRNN/PeMS04/DCRNN_UniNoGc.py -g 0 &
python main/train.py -c model_setting/compare_method/DCRNN/PeMS04/DCRNN_RanNoGc.py -g 1  