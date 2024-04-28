cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/NLinear/PeMS04/NLinear_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/PeMS04/NLinear_LN_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/NLinear/PeMS04/NLinear_SLN_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/PeMS04/NLinear_TLN_Short.py -g 0

