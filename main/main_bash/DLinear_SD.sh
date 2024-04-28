cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_Long.py -g 1 & 
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_LN_Long.py -g 1 &
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_SLN_Long.py -g 1 & 
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_TLN_Long.py -g 1 &

python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_Short.py -g 1 & 
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_LN_Short.py -g 1 &
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_SLN_Short.py -g 1 & 
python main/train.py -c model_setting/compare_method/DLinear/SD/DLinear_TLN_Short.py -g 1

