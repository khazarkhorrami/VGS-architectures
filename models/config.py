
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "../../../features/SPOKEN-COCO/", #"/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/", #      
  "feature_path_MSCOCO": "../../../features/MSCOCO/", #  "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/MSCOCO/", # 
  "json_path_SPOKENCOCO" : "../../../data/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/apc/wavenet_l5_f500/",
}


action_parameters = {
  "use_pretrained": False,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":False,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : True,
  "number_of_epochs" : 30,
  "chunk_length":1000
}

model_settings = {
    "model_name": "CNN0",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
