cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/GWN/PeMS04/GWN_INoAda_HypernFC.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/GWN_INoAda_HypernS.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/GWN_INoAda_HypernT.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/GWN_INoAda_HypernWb.py -g 1 &
