cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/hyper_method/GWN/METR/GWN_INoAda_Hyper.py -g 0 &
python main/train.py -c model_setting/hyper_method/GWN/METR/GWN_ONoAda_Hyper.py -g 0