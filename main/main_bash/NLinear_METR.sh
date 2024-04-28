cd /home/fangshen/PromptNet/
source activate st_pre

python main/train.py -c model_setting/compare_method/NLinear/METR_LA/NLinear_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/METR_LA/NLinear_LN_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/NLinear/METR_LA/NLinear_SLN_Short.py -g 0 & 
python main/train.py -c model_setting/compare_method/NLinear/METR_LA/NLinear_TLN_Short.py -g 0

