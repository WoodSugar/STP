cd /home/fangshen/PromptNet/
source activate st_pre


# python main/train.py -c model_setting/compare_method/LR/SD/LR_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/SD/LR_LN_Long.py -g 0 & 
# python main/train.py -c model_setting/compare_method/LR/SD/LR_TLN_Long.py -g 0 &
# python main/train.py -c model_setting/compare_method/LR/SD/LR_SLN_Long.py -g 0 &

python main/train.py -c model_setting/hyper_method/MLP/SD/MLP_HypernFC.py -g 0 & 
python main/train.py -c model_setting/hyper_method/MLP/SD/MLP_HypernS.py -g 0 &
python main/train.py -c model_setting/hyper_method/MLP/SD/MLP_HypernT.py -g 0 &
python main/train.py -c model_setting/hyper_method/MLP/SD/MLP_HypernWb.py -g 0 
