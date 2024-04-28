#!/bin/bash

cd /home/fangshen/ST_PreTrain/
source activate st_pre


# ========================================================== Dataset 2 Bus ==========================================================
# Bus LR long
# python main/test.py -c model_setting/compare_method/LR_MyBus.py -ck ~/data/Forecast_Result/LR_MyBus_500/165b831b3b0fc8f8437e3e94a18062dd/LR_best_val_MAE.pt -g 1
# Bus LR Short
# python main/test.py -c model_setting/compare_method/LR_MyBusShort.py -ck ~/data/Forecast_Result/LR_MyBus_500/75e2452c568cbd2e28feb37093dcaad2/LR_best_val_MAE.pt -g 1

# Bus MLP
# python main/test.py -c model_setting/compare_method/MLP_MyBus.py -ck ~/data/Forecast_Result/MLP_MyBus_500/14f54cf52c5a67040fa7b4144b3fdcc0/MLP_best_val_MAE.pt -g 1

# Bus DCRNN
# python main/test.py -c model_setting/compare_method/DCRNN_MyBus.py -ck ~/data/Forecast_Result/DCRNN_MyBus_500/3d7de9fba4d284cd535d65ec8be41e65/DCRNN_best_val_MAE.pt -g 1

# Bus GWN
# python main/test.py -c ~/data/Forecast_Result/GraphWaveNet_MyBus_500/20eec96901d90739951fc7d15e963833/GWNet_MyBus.py -ck ~/data/Forecast_Result/GraphWaveNet_MyBus_500/20eec96901d90739951fc7d15e963833/GraphWaveNet_best_val_MAE.pt -g 1
# python main/test.py -c model_setting/compare_method/GWNet_MyBus_93cf17.py -ck ~/data/Forecast_Result/GraphWaveNet_MyBus_500/93cf1750ca7a3960d2001bacd065623b/GraphWaveNet_best_val_MAE.pt -g 1

# Bus MTGNN
# python main/test.py -c model_setting/compare_method/MTGNN_MyBus.py -ck ~/data/Forecast_Result/MTGNN_MyBus_500/4c04e5be33d6fbbdced1453efa66bfd5/MTGNN_best_val_MAE.pt -g 1

# Bus STSGCN
# python main/test.py -c model_setting/compare_method/STSGCN_MyBus_01d2b5.py -ck ~/data/Forecast_Result/STSGCN_MyBus_500/01d2b53732f267fc909ec32dc149a31e/STSGCN_best_val_MAE.pt -g 1

# python main/test.py -c model_setting/compare_method/STSGCN_MyBus.py -ck ~/data/Forecast_Result/STSGCN_MyBus_500/9c05e86c5f7dfe96f3e339a89af103b1/STSGCN_best_val_MAE.pt -g 2

# Bus STFGNN
# python main/test.py -c model_setting/compare_method/STFGNN_MyBus_1af0af.py -ck ~/data/Forecast_Result/STFGNN_MyBus_500/1af0af8ffeb8f8ddb8a31c857f59f6b2/STFGNN_best_val_MAE.pt -g 1

# Bus SubwayPretrain+MLPHeader
# python main/test.py -c model_setting/MLP_EncoderUnmaks2TransformerDecoder_MyBus.py -ck ~/data/Predictor_Result/Predictor_EncoderUnmask2TransformerDecoder_MyBus_500/14ca165ca21c13d5e1f9975351ec7cf9/Predictor_EncoderUnmask2TransformerDecoder_best_val_MAE.pt -g 2


# ========================================================== Dataset 3 Taxi ==========================================================

# Taxi finetune1
# python main/test.py -c model_setting/fine_tune/MyTaxi_STConv_EncoderUnmaks2TransformerDecoder_f18112.py -ck ~/data/Predictor_Result/STConvEncoderUnmask2TransformerDecoder_MyTaxi_500/f181121afc1923ef7c098f3fc0c05697/STConvEncoderUnmask2TransformerDecoder_best_val_MAE.pt -g 1

# Taxi finetune2
# python main/test.py -c model_setting/fine_tune/MyTaxi_STConv_EncoderUnmaks2TransformerDecoder_65c642.py -ck ~/data/Predictor_Result/STConvEncoderUnmask2TransformerDecoder_MyTaxi_500/65c642bc3a29da053ce10c422e62ca59/STConvEncoderUnmask2TransformerDecoder_best_val_MAE.pt -g 1

# Taxi finetune3
# python main/test.py -c model_setting/fine_tune/MyTaxi_STConv_EncoderUnmaks2TransformerDecoder_1e9af7.py -ck ~/data/Predictor_Result/STConvEncoderUnmask2TransformerDecoder_MyTaxi_500/1e9af70f2ab540f1dd63b633a7774e74/STConvEncoderUnmask2TransformerDecoder_best_val_MAE.pt -g 1

# Taxi finetune4
# python main/test.py -c model_setting/fine_tune/MyTaxi_STConv_EncoderUnmaks2TransformerDecoder_03e6f4.py -ck ~/data/Predictor_Result/STConvEncoderUnmask2TransformerDecoder_MyTaxi_500/03e6f458ce02115defbf30be99b084e2/STConvEncoderUnmask2TransformerDecoder_best_val_MAE.pt -g 1


# Taxi LR short
# python main/test.py -c model_setting/compare_method/LR_MyTaxiShort.py -ck ~/data/Forecast_Result/LR_MyTaxi_500/c8d73bd7d0ffdb7488fe604d9a4c772a/LR_best_val_MAE.pt

# Taxi LR long
# python main/test.py -c model_setting/compare_method/LR_MyTaxi.py -ck ~/data/Forecast_Result/LR_MyTaxi_500/24ff85ede7cc734f80c56a96b940775e/LR_best_val_MAE.pt

# Taxi MLP 
python main/test.py -c model_setting/compare_method/MLP_Taxi.py -ck ~/data/Forecast_Result/MLP_MyTaxi_500/6ea6e4a638251be67d593d779c769790/MLP_best_val_MAE.pt

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
# python main/test.py -c model_setting/compare_method/Crossformer_MyTaxi.py -ck ~/data/Forecast_Result/Crossformer_MyTaxi_500/35fbf820946a584fc85f1ecbf4116d20/Crossformer_best_val_MAE.pt

# Taxi CrossformerLong
# python main/test.py -c model_setting/compare_method/Crossformer_MyTaxiLong.py -ck ~/data/Forecast_Result/Crossformer_MyTaxi_500/735d4eb1dde13b7d62dff9e9107ce335/Crossformer_best_val_MAE.pt

# Taxi STAEformer
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxi.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/9460f8aa1d11794a397f13ef06e3cf6c/STAEformer_best_val_MAE.pt
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxi.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/3d2ae90a95adc5f5759cb04dfffee807/STAEformer_best_val_MAE.pt
# python main/test.py -c model_setting/compare_method/STAEformer_MyTaxid99.py -ck ~/data/Forecast_Result/STAEformer_MyTaxi_500/d99fc3ddc1b28d1f92cdb8a5c78e4f37/STAEformer_best_val_MAE.pt


# ========================================================== Dataset 4 PeMS03 ==========================================================

