cd /home/fangshen/PromptNet/
source activate st_pre


# python main/train.py -c model_setting/compare_method/LR/Subway/LR_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/Subway/LR_LN_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/Subway/LR_TLN_Long.py -g 0 &
# python main/train.py -c model_setting/compare_method/LR/Subway/LR_SLN_Long.py -g 0 &

python main/train.py -c model_setting/compare_method/LR/Subway/LR_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/LR/Subway/LR_LN_Short.py -g 0 &
# python main/train.py -c model_setting/compare_method/LR/Subway/LR_TLN_Short.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/Subway/LR_SLN_Short.py -g 0 & 

