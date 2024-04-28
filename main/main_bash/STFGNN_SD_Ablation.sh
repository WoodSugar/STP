cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/STFGNN/SD/STFGNN_I_HypernFC.py -g 1 & 
python main/train.py -c model_setting/hyper_method/STFGNN/SD/STFGNN_I_HypernS.py -g 1 &
python main/train.py -c model_setting/hyper_method/STFGNN/SD/STFGNN_I_HypernT.py -g 1 

