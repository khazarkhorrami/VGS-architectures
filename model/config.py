
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "../../../features/SPOKEN-COCO/", #"/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/", #      
  "feature_path_MSCOCO": "../../../features/MSCOCO/", #  "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/MSCOCO/", # 
  "json_path_SPOKENCOCO" : "../../../data/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "models_dir": "../../models/",
}


action_parameters = {
  "use_pretrained": True,
  "training_mode": False,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : False,
  "number_of_epochs" : 30,
  "chunk_length":10000
}

model_settings = {
    "model_name": "CNN0",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
