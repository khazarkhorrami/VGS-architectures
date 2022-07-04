#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The parent VGS class

all other action classes are children of VGS

"""


from data_preprocessing import prepare_XY , read_feature_filenames,  expand_feature_filenames2, prepareX_apc

class Data:
    
    def __init__(self, dataset_name, chunk_length, feature_path_SPOKENCOCO, feature_path_MSCOCO, json_path_SPOKENCOCO):
        self.dataset_name = dataset_name
        self.chunk_length = chunk_length
        self.feature_path_SPOKENCOCO = feature_path_SPOKENCOCO
        self.feature_path_MSCOCO = feature_path_MSCOCO
        self.json_path_SPOKENCOCO = json_path_SPOKENCOCO

    def set_feature_paths (self):
        if self.dataset_name == "SPOKEN-COCO":
            self.feature_path_audio = self.feature_path_SPOKENCOCO 
            self.feature_path_image = self.feature_path_MSCOCO     
    
    def prepare_feature_names (self, split):
        if self.dataset_name == "SPOKEN-COCO":
            self.feature_names = read_feature_filenames (self.json_path_SPOKENCOCO, split , shuffle_data = True )  
        
    def prepare_chunked_names (self , captionID):
        n = len(self.feature_names)
        Ynames_all, Xnames_all , Znames_all = [],[],[]       
        for start_chunk in range(0, n , self.chunk_length):
            end_chunk = start_chunk + self.chunk_length
            chunk = self.feature_names [start_chunk:end_chunk ]            
            Ynames, Xnames , Znames = expand_feature_filenames2(chunk , captionID)
            Ynames_all.append(Ynames)
            Xnames_all.append(Xnames)
            Znames_all.append(Znames)
        return Ynames_all, Xnames_all , Znames_all



    def prepare_data_names(self , split, captionID):
        self.set_feature_paths()
        self.prepare_feature_names(split)  
        Ynames_all, Xnames_all , Znames_all = self.prepare_chunked_names(captionID)        
        return Ynames_all, Xnames_all , Znames_all
    
    

            
    def load_data(self, Ynames, Xnames , Znames):
       
        Ydata, Xdata = prepare_XY (Ynames, Xnames , self.feature_path_audio , self.feature_path_image , self.length_sequence )

        return Ydata, Xdata
    

        
        
