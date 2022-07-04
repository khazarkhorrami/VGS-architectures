
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/", #  "../../features/SPOKEN-COCO/",    
  "feature_path_MSCOCO":  "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/MSCOCO/", # "../../features/MSCOCO/",
  "json_path_SPOKENCOCO" : "../../data/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/apc/wavenet_l5_f500/",
}


action_parameters = {
  "use_pretrained": True,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : True,
  "number_of_epochs" : 30,
  "chunk_length":10000
}

feature_settings = {
    "model_name": "CNNatt_dual",
    "model_subname": "v0",
    "length_sequence" : 200,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
