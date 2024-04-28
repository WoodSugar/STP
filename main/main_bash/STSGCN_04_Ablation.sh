cd /home/fangshen/PromptNet/
source activate st_pre


# python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/HyperSTSGCNnFC.py -g 1 & 
# python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/HyperSTSGCNnS.py -g 1 &
# python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/HyperSTSGCNnT.py -g 1

python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/STSGCN_I_HypernFC.py -g 0 & 
python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/STSGCN_I_HypernS.py -g 0 &
python main/train.py -c model_setting/hyper_method/STSGCN/PeMS04/STSGCN_I_HypernT.py -g 0