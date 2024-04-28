cd /home/fangshen/PromptNet/
source activate st_pre

# python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/STGCN_I_HypernFC.py -g 0 &
# python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/STGCN_I_HypernS.py  -g 0 & 
# python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/STGCN_I_HypernT.py  -g 0 &
# python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/STGCN_I_HypernWb.py -g 0

python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/HyperSTGCNnFC.py -g 0 &
python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/HyperSTGCNnS.py -g 0 &
python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/HyperSTGCNnT.py -g 0 &
python main/train.py -c model_setting/hyper_method/STGCN/PeMS04/HyperSTGCNnWb.py -g 0