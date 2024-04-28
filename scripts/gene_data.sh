# cd /home/fangshen/PromptNet/
source activate prompt_net


# ==== PeMS04New ====
echo "==== PeMS04New ===="
python scripts/data_preparation/PeMS04New/generate_training_data.py --pre_train True
python scripts/data_preparation/PeMS04New/generate_training_data.py --pre_train False


# ==== PeMS07New ====
echo "==== PeMS07New ===="
python scripts/data_preparation/PeMS07New/generate_training_data.py --pre_train True
python scripts/data_preparation/PeMS07New/generate_training_data.py --pre_train False


# ==== PeMS08New ====
echo "==== PeMS08New ===="
python scripts/data_preparation/PeMS08New/generate_training_data.py --pre_train True
python scripts/data_preparation/PeMS08New/generate_training_data.py --pre_train False


# ==== METR_LA ====
echo "==== METR_LA ===="
python scripts/data_preparation/METR_LA/generate_training_data.py --pre_train True
python scripts/data_preparation/METR_LA/generate_training_data.py --pre_train False


# ==== LargeST_SD ====
echo "==== LargeST_SD ===="
python scripts/data_preparation/LargeST_SD/generate_training_data.py --pre_train True
python scripts/data_preparation/LargeST_SD/generate_training_data.py --pre_train False