cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_IdnNoAda.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_UniNoAda.py -g 1 &
python main/train.py -c model_setting/compare_method/GWN/PeMS04/GWN_RanNoAda.py -g 1