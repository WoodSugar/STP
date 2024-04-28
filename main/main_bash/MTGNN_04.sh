cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/MTGNN/PeMS04/MTGNN_Ori.py -g 0 &
python main/train.py -c model_setting/compare_method/MTGNN/PeMS04/MTGNN_Idn.py -g 0 &

python main/train.py -c model_setting/compare_method/MTGNN/PeMS04/MTGNN_Uni.py -g 0 &
python main/train.py -c model_setting/compare_method/MTGNN/PeMS04/MTGNN_Ran.py -g 0