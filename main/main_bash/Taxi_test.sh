#!/bin/bash

cd /home/fangshen/ST_PreTrain/
source activate st_pre


# ========================================================== Dataset 3 Taxi ==========================================================
# Taxi LR short
# python main/test.py -c model_setting/compare_method/LR_MyTaxiShort.py -ck ~/data/Forecast_Result/LR_MyTaxi_500/c8d73bd7d0ffdb7488fe604d9a4c772a/LR_best_val_MAE.pt

# Taxi LR long
# python main/test.py -c model_setting/compare_method/LR_MyTaxi.py -ck ~/data/Forecast_Result/LR_MyTaxi_500/24ff85ede7cc734f80c56a96b940775e/LR_best_val_MAE.pt

# Taxi MLP 
# # python main/test.py -c model_setting/compare_method/ .py -ck ~/data/Forecast_Result/MLP_MyTaxi_500/6ea6e4a638251be67d593d779c769790/MLP_best_val_MAE.pt
# python main/test.py -c model_setting/compare_method/MLP_TaxiShort.py -ck ~/data/Forecast_Result/MLPShort_MyTaxi_500/166a354eb71af2afd0c4e483c9840b5d/MLPShort_best_val_MAE.pt

# Taxi DCRNN
# python main/test.py -c model_setting/compare_method/DCRNN_MyTaxi.py -ck ~/data/Forecast_Result/DCRNN_MyTaxi_500/8b6e7c59cb819908e67dc3fb52bde3f8/DCRNN_best_val_MAE.pt

# Taxi GWNet
# python main/test.py -c model_setting/compare_method/GWNet_MyTaxi.py -ck ~/data/Forecast_Result/GraphWaveNet_MyTaxi_500/ce957d43eba7d9a4fc9806c79309c888/GraphWaveNet_best_val_MAE.pt

# Taxi MTGNN
# python main/test.py -c model_setting/compare_method/MTGNN_MyTaxi.py -ck ~/data/Forecast_Result/MTGNN_MyTaxi_500/90f1ce026e8a4ce737154af83fbfac05/MTGNN_best_val_MAE.pt

# Taxi STSGCN
# python main/test.py -c model_setting/compare_method/STSGCN_MyTaxi.py -ck ~/data/Forecast_Result/STSGCN_MyTaxi_500/87e5673123ece4d475bf72d7564cf70c/STSGCN_best_val_MAE.pt

# Taxi STFGNN
# python main/test.py -c model_setting/compare_method/STFGNN_MyTaxi.py -ck ~/data/Forecast_Result/STFGNN_MyTaxi_500/d401f2b0bf313472c0ccda8c6bc8fdb9/STFGNN_best_val_MAE.pt

# Taxi Crossformer
# # python main/test.py -c model_setting/compare_method/Crossformer_MyTaxi.py -ck ~/data/Forecast_Result/Crossformer_MyTaxi_500/35fbf820946a584fc85f1ecbf4116d20/Crossformer_best_val_MAE.pt
python main/test.py -c model_setting/compare_method/Crossformer_MyTaxiLong2LowEn.py -ck ~/data/Forecast_Result/Crossformer_MyTaxi_500/322a5d8ce75cc891d9f1d3c77854eafb/Crossformer_best_val_MAE.pt

# Taxi CrossformerLong
# python main/test.py -c model_setting/compare_method/Crossformer_MyTaxiLong.py -ck ~/data/Forecast_Result/Crossformer_MyTaxi_500/735d4eb1dde13b7d62dff9e9107ce335/Crossformer_best_val_MAE.pt

# Taxi STAEformer
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxi.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/9460f8aa1d11794a397f13ef06e3cf6c/STAEformer_best_val_MAE.pt
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxi.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/3d2ae90a95adc5f5759cb04dfffee807/STAEformer_best_val_MAE.pt
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxid99.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/d99fc3ddc1b28d1f92cdb8a5c78e4f37/STAEformer_best_val_MAE.pt

# Taxi 

# # python  main/test.py -c model_setting/fine_tune/Taxi_EnPropose_DeTAttn.py -ck ~/data/Finetune_Result/TaxiProposeMSE_TAttn_MyTaxi_500/35d939f17b87eb0b8266b22b07b50404/TaxiProposeMSE_TAttn_best_val_MAE.pt

# Taxi SConv3
# python main/test.py -c model_setting/fine_tune/TaxiEnPropose_DeTMetaConvSConv.py -ck ~/data/Finetune_Result/TaxiEnPropose_DeTMetaConvSConv_MyTaxi_500/6cafd66923254fdef27cb90c21566cab/TaxiEnPropose_DeTMetaConvSConv_best_val_MAE.pt

# Taxi SConv1
python main/test.py -c model_setting/fine_tune/TaxiEnPropose_DeTMetaConvSConvLowD.py -ck ~/data/Finetune_Result/TaxiEnPropose_DeTMetaConvSConvLowD_MyTaxi_500/bac0fe42a837f1c84800c445a4fab5ac/TaxiEnPropose_DeTMetaConvSConvLowD_best_val_MAE.pt


# TaxiEnPropose_DeSTConvLowD
python main/test.py -c model_setting/fine_tune/TaxiEnPropose_DeSTConvLowD.py -ck ~/data/Finetune_Result/TaxiEnPropose_DeSTConv_MyTaxi_500/7d149095ab23bdbe19cd8e0ebd16443f/TaxiEnPropose_DeSTConv_best_val_MAE.pt

# TaxiEnPropose_DeSTConv
python main/test.py -c model_setting/fine_tune/TaxiEnPropose_DeSTConv.py -ck /data/fangshen/Finetune_Result/TaxiEnPropose_DeSTConv_MyTaxi_500/5496ff58061c52d22a511a69ba609737/TaxiEnPropose_DeSTConv_best_val_MAE.pt