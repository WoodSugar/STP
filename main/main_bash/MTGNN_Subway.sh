cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/MTGNN/Subway/MTGNN_Ori.py -g 1 &
python main/train.py -c model_setting/compare_method/MTGNN/Subway/MTGNN_Idn.py -g 1 &

python main/train.py -c model_setting/compare_method/MTGNN/Subway/MTGNN_Uni.py -g 1 &
python main/train.py -c model_setting/compare_method/MTGNN/Subway/MTGNN_Ran.py -g 1