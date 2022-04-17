
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/" , #"../../features/SPOKEN-COCO/",
  "feature_path_MSCOCO": "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/MSCOCO/" , #"../../features/MSCOCO/",
  "json_path_SPOKENCOCO" : "../../data/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/model00/",
}


action_parameters = {
  "use_pretrained": False,
  "training_mode": False,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : True,
  "save_best_loss" : False,
  "find_recall" : True,
  "number_of_epochs" : 100,
  "chunk_length":10000
}

feature_settings = {
    "model_name": "CNNatt",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
