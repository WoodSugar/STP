cd /home/fangshen/PromptNet/
source activate st_pre


python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_METRLong.py -g 0
python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_METRShort.py -g 0

python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_LN_METRLong.py -g 0
python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_LN_METRShort.py -g 0

python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_SLN_METRLong.py -g 0
python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_SLN_METRShort.py -g 0

python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_TLN_METRLong.py -g 0
python main/train.py -c model_setting/compare_method/MLP/METR_LA/MLP_TLN_METRShort.py -g 0
