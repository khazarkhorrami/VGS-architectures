

from tensorflow import keras
from loss import  prepare_triplet_data ,  triplet_loss , prepare_MMS_data , mms_loss_normal, mms_loss_hard
from recall import calculate_recallat10
from features import prepare_XY , read_feature_filenames,  expand_feature_filenames2, prepareX_apc

class VGS:
    
    def __init__(self):
        pass
    
    def compile_model (self, vgs_model, loss):
        if loss == "MMS":      
            vgs_model.compile(loss = mms_loss_normal, optimizer= keras.optimizers.Adam(lr=1e-03))         
        elif loss == "Triplet":
            vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
        print(vgs_model.summary())
       
    def prepare_data(self, Ynames, Xnames , Znames, feature_path_audio , feature_path_image , length_sequence , normalize ,loss ):      
        Ydata, Xdata = prepare_XY (Ynames, Xnames , feature_path_audio , feature_path_image , length_sequence , normalize)
        if loss == "MMS":
            [Yin, Xin] , bin_target = prepare_MMS_data (Ydata, Xdata , shuffle_data = False)
        if loss == "Triplet":                   
            Yin, Xin, bin_target = prepare_triplet_data (Ydata, Xdata)
        return Yin, Xin, bin_target
    
    def train_model(self, vgs_model, Yin, Xin , bin_target):            
        history = vgs_model.fit([Yin, Xin ], bin_target, shuffle=False, epochs=1,batch_size=120)                      
        return history.history['loss'][0]
     
    def validate_model (self, vgs_model, Yin, Xin , bin_target):
        evaluation_loss = vgs_model.evaluate([Yin, Xin ], bin_target, batch_size=120)
        return evaluation_loss
    
    def test_model (self,vgs_model, Yin, Xin , bin_target):
        preds = self.vgs_model.predict([Yin, Xin ])
        return preds
    
    def iterate_and_train(self, vgs_model, Ynames_all, Xnames_all , Znames_all , feature_path_audio , feature_path_image , length_sequence , normalize, loss):
        number_of_chunks = len(Ynames_all)
        train_loss_all_chunks = 0
        for counter_chunk in range(number_of_chunks): 
        
            Ynames = Ynames_all [counter_chunk]
            Xnames = Xnames_all [counter_chunk]
            Znames = Znames_all [counter_chunk]
            Yin, Xin, bin_target = self.prepare_data(Ynames, Xnames , Znames, feature_path_audio , feature_path_image , length_sequence , normalize , loss)
            train_loss_chunk = self.train_model(vgs_model, Yin, Xin , bin_target)
            train_loss_all_chunks += train_loss_chunk
            
        train_loss = train_loss_all_chunks / number_of_chunks
        return train_loss
    
    def iterate_and_validate(self, vgs_model, Ynames_all, Xnames_all , Znames_all , feature_path_audio , feature_path_image , length_sequence, normalize , loss):
        number_of_chunks = len(Ynames_all)
        evaluation_loss_all_chunks = 0
        for counter_chunk in range(number_of_chunks): 
        
            Ynames = Ynames_all [counter_chunk]
            Xnames = Xnames_all [counter_chunk]
            Znames = Znames_all [counter_chunk]
            Yin, Xin, bin_target = self.prepare_data(Ynames, Xnames , Znames, feature_path_audio , feature_path_image , length_sequence, normalize , loss)
            evaluation_loss_chunk = self.validate_model(vgs_model, Yin, Xin , bin_target)
            evaluation_loss_all_chunks += evaluation_loss_chunk
            
        evaluation_loss = evaluation_loss_all_chunks / number_of_chunks
        return evaluation_loss