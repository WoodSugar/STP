cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/GWN/SD/GWN_INoAda_Hyper.py -g 1 &
python main/train.py -c model_setting/hyper_method/GWN/SD/GWN_ONoAda_Hyper.py -g 1