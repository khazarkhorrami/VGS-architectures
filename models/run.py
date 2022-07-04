#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:08:40 2022

@author: hxkhkh
"""

import config as cfg
from data import Data

class RUN_experiment():
    
    def __init__(self):
        
        # paths
        self.feature_path_SPOKENCOCO = cfg.paths['feature_path_SPOKENCOCO']
        self.feature_path_MSCOCO = cfg.paths['feature_path_MSCOCO']
        self.json_path_SPOKENCOCO = cfg.paths["json_path_SPOKENCOCO"]
        self.dataset_name = cfg.paths['dataset_name']
        self.model_dir = cfg.paths['modeldir']
        
        # action parameters
        self.use_pretrained = cfg.action_parameters['use_pretrained']
        self.training_mode = cfg.action_parameters['training_mode']
        self.evaluating_mode = cfg.action_parameters['evaluating_mode']
        self.saving_mode = cfg.action_parameters['save_model']
        self.save_best_recall = cfg.action_parameters['save_best_recall']
        self.save_best_loss = cfg.action_parameters['save_best_loss']
        self.find_recall = cfg.action_parameters['find_recall']
        self.number_of_epochs = cfg.action_parameters['number_of_epochs']
        self.chunk_length = cfg.action_parameters['chunk_length']
        
        # model setting
        self.model_name = cfg.feature_settings ['model_name']
        self.model_subname = cfg.feature_settings ['model_subname']
        self.length_sequence = cfg.feature_settings['length_sequence']
        self.Xshape = cfg.feature_settings['Xshape']
        self.Yshape = cfg.feature_settings['Yshape']
        self.input_dim = [self.Xshape,self.Yshape] 
        self.loss = "MMS"
        self.temperature = 0.1
        self.epoch_counter = 0
        
        #self.length_sequence = self.Xshape[0]
        self.split = 'train'
        self.captionID = 0
        self.feature_names = []
        
        
        self.data = Data(self.dataset_name, self.chunk_length, self.feature_path_SPOKENCOCO, self.feature_path_MSCOCO, self.json_path_SPOKENCOCO)

    def iterate_data(self, split, captionID):
        
        Ynames_all, Xnames_all , Znames_all = self.data.prepare_data_names(self.split, self.captionID)
        self.number_of_chunks = len(Ynames_all)
        
        for counter_chunk in range(self.number_of_chunks): 
        
            Ynames = Ynames_all [counter_chunk]
            Xnames = Xnames_all [counter_chunk]
            Znames = Znames_all [counter_chunk]
        
        return Ynames, Xnames, Znames
    
    def prepare_model (self):
        pass
    
    def train_model(self):
        #vgs.train_model()
        pass
    
    def validate_model (self):
        pass
    
    def test_model(self):
        pass
    
    # def __call__(self):
    
    #     self.define_and_compile_models()
  
    #     initialized_output = self.initialize_model_parameters()
        
    #     if self.use_pretrained:
    #         self.vgs_model.load_weights(self.model_dir + 'model_weights.h5')

    #     for epoch_counter in numpy.arange(self.number_of_epochs):
    #         self.epoch_counter = epoch_counter
    #         print('......... epoch ...........' , str(epoch_counter))
            
    #         if self.training_mode:           
    #             training_output = self.train_model()           
                    
    #         else:
    #             training_output = 0
                
    #         if self.evaluating_mode:
    #             validation_output = self.evaluate_model()
    #         else: 
    #             validation_output = [0, 0 , 0 ]
                
    #         if self.saving_mode:
    #             self.save_model(initialized_output, training_output, validation_output)    

run_object = RUN_experiment()
Ynames, Xnames, Znames = run_object.iterate_data(split = 'train', captionID = 0)