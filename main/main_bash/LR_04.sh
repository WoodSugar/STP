cd /home/fangshen/PromptNet/
source activate st_pre


# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_LN_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_TLN_Long.py -g 0 &
# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_SLN_Long.py -g 0 &

python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_Short.py -g 0 &
python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_LN_Short.py -g 0 &
# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_TLN_Short.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/PeMS04/LR_SLN_Short.py -g 0 & 

