

import config as cfg
from data import Data
from networks import CNN0, CNNatt, AVLnet
from models import VGS


import os
import numpy
import scipy.io
from matplotlib import pyplot as plt


class RUN_vgs():
    
    def __init__(self):
        
        # paths
        self.feature_path_SPOKENCOCO = cfg.paths['feature_path_SPOKENCOCO']
        self.feature_path_MSCOCO = cfg.paths['feature_path_MSCOCO']
        self.json_path_SPOKENCOCO = cfg.paths["json_path_SPOKENCOCO"]
        self.dataset_name = cfg.paths['dataset_name']
        self.models_dir = cfg.paths['models_dir']
        
        
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
        self.model_name = cfg.model_settings ['model_name']
        self.model_subname = cfg.model_settings ['model_subname']
        self.length_sequence = cfg.model_settings['length_sequence']
        self.Xshape = cfg.model_settings['Xshape']
        self.Yshape = cfg.model_settings['Yshape']
        self.input_dim = [self.Xshape,self.Yshape] 
        self.loss = "Triplet"
        self.temperature = 0.1
        self.epoch_counter = 0
        
        #self.length_sequence = self.Xshape[0]
        self.split = 'train'
        self.captionID = 0
        self.feature_names = []
        self.normalize = True      
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        

        self.current_data = Data(self.dataset_name, self.chunk_length, self.feature_path_SPOKENCOCO, self.feature_path_MSCOCO, self.json_path_SPOKENCOCO)
        
        if self.model_name =="CNN0":
            self.current_model = CNN0(self.input_dim)
        elif self.model_name =="CNNatt":
            self.current_model = CNNatt(self.input_dim)
        elif self.model_name == "AVLnet":
            self.current_model = AVLnet(self.input_dim)   
        
        self.vgs = VGS()
        
        
    def set_feature_paths (self):
        if self.dataset_name == "SPOKEN-COCO":
            self.feature_path_audio = self.feature_path_SPOKENCOCO 
            self.feature_path_image = self.feature_path_MSCOCO 
            
    def initialize_model_parameters(self):
        
        if self.use_pretrained:
            data = scipy.io.loadmat(os.path.join (self.model_dir , 'valtrainloss.mat'), variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
            allepochs_valloss = data['allepochs_valloss'][0]
            allepochs_trainloss = data['allepochs_trainloss'][0]
            all_avRecalls = data['all_avRecalls']#[0]
            all_vaRecalls = data['all_vaRecalls']#[0]
            
            allepochs_valloss = numpy.ndarray.tolist(allepochs_valloss)
            allepochs_trainloss = numpy.ndarray.tolist(allepochs_trainloss)
            all_avRecalls = numpy.ndarray.tolist(all_avRecalls)
            all_vaRecalls = numpy.ndarray.tolist(all_vaRecalls)
            recall_indicator = numpy.max(allepochs_valloss)
            val_indicator = numpy.min(allepochs_valloss)
        else:
            allepochs_valloss = []
            allepochs_trainloss = []
            all_avRecalls = []
            all_vaRecalls = []
            recall_indicator = 0
            val_indicator = 1000

        saving_params = [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ]       
        return saving_params
    
    def save_model(self, vgs_model , initialized_output , training_output, validation_output):
        
        os.makedirs(self.model_dir, exist_ok=1)
        [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ] = initialized_output
        [final_recall_av, final_recall_va , final_valloss] = validation_output 
        [epoch_recall_av, epoch_recall_va , epoch_valloss] = validation_output              
            
        if self.save_best_recall:
            epoch_recall = ( epoch_recall_av + epoch_recall_va ) / 2
            if epoch_recall >= recall_indicator: 
                recall_indicator = epoch_recall
                vgs_model.save_weights(os.path.join(self.model_dir, 'model_weights.h5'))#'%smodel_weights.h5' % self.model_dir)
        else :
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                vgs_model.save_weights(os.path.join(self.model_dir, 'model_weights.h5'))
                      
        allepochs_trainloss.append(training_output)  
        allepochs_valloss.append(epoch_valloss)
        if self.find_recall: 
            all_avRecalls.append(epoch_recall_av)
            all_vaRecalls.append(epoch_recall_va)
        save_file = os.path.join(self.model_dir , 'valtrainloss.mat')
        scipy.io.savemat(save_file, 
                          {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })  
        
        self.make_plot( [allepochs_trainloss, allepochs_valloss , all_avRecalls, all_vaRecalls ])

        
    def make_plot (self, plot_lists):       
        plt.figure()
        plot_names = ['training loss','validation loss','speech_to_image recall','image_to_speech recall']
        for plot_counter, plot_value in enumerate(plot_lists):
            plt.subplot(2,2,plot_counter+1)
            plt.plot(plot_value)
            plt.ylabel(plot_names[plot_counter])
            plt.grid()

        plt.savefig(os.path.join(self.model_dir, 'evaluation_plot.pdf'), format = 'pdf')


    def train_vgs(self, vgs_model):
        self.split = 'train'
        Ynames_all, Xnames_all , Znames_all = self.current_data.prepare_data_names(self.split, self.captionID)
        train_loss = self.vgs.iterate_and_train(vgs_model, Ynames_all, Xnames_all , Znames_all , self.feature_path_audio , self.feature_path_image , self.length_sequence , self.normalize, self.loss)
        return train_loss

    
    def evaluate_vgs(self, vgs_model): 
        self.split = 'val'
        Ynames_all, Xnames_all , Znames_all = self.current_data.prepare_data_names(self.split, self.captionID)
        val_loss = self.vgs.iterate_and_validate(vgs_model, Ynames_all, Xnames_all , Znames_all , self.feature_path_audio , self.feature_path_image , self.length_sequence, self.normalize , self.loss)
        return [0,0,val_loss]
        
    def __call__(self):
        
        self.set_feature_paths()      
        initialized_output = self.initialize_model_parameters()   
        
        vgs_model, visual_embedding_model, audio_embedding_model = self.current_model.build_model() 
        self.vgs.compile_model (vgs_model, self.loss)
        if self.use_pretrained:
            vgs_model.load_weights(os.path.join (self.model_dir , 'model_weights.h5'))

        for epoch_counter in numpy.arange(self.number_of_epochs):
            self.epoch_counter = epoch_counter
            print('......... epoch ...........' , str(epoch_counter))
            
            if self.training_mode:           
                training_output = self.train_vgs(vgs_model)           
                    
            else:
                training_output = 0
                
            if self.evaluating_mode:
                validation_output = self.evaluate_vgs(vgs_model)
            else: 
                validation_output = [0, 0 , 0 ]
                
            if self.saving_mode:
                self.save_model(vgs_model, initialized_output, training_output, validation_output)    

run_object = RUN_vgs()
run_object()