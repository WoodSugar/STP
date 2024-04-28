cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_Long.py -g 0 & 
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_LN_Long.py -g 0 &
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_SLN_Long.py -g 0 & 
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_TLN_Long.py -g 0 &

python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_LN_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_SLN_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/DLinear/PeMS03/DLinear_TLN_Short.py -g 0

