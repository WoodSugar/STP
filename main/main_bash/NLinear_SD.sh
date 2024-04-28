cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_Long.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_LN_Long.py -g 0 &
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_SLN_Long.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_TLN_Long.py -g 0 &

python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_LN_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_SLN_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/SD/NLinear_TLN_Short.py -g 0

