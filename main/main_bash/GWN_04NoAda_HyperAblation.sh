cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/GWN/PeMS04/HyperGWNNoAdanFC.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/HyperGWNNoAdanS.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/HyperGWNNoAdanT.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/PeMS04/HyperGWNNoAdanWb.py -g 1
