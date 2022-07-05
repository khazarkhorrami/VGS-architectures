
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import  MaxPooling1D,  Conv1D,Conv2D
from tensorflow.keras.layers import Softmax, Permute, AveragePooling1D, Concatenate

from vgs import VGS

class CNN0(VGS):
    
    def __init__(self, model_subname, input_dim):
        
        self.model_subname = model_subname
        self.input_dim = input_dim
        
    def build_audio_model (self ):
        
        [Xshape, Yshape] = self.input_dim
        dropout_size = 0.3
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape)
          
        forward1 = Conv1D(128,5,padding="same",activation=activation_C,name = 'conv1')(audio_sequence)
        dr1 = Dropout(dropout_size)(forward1)
        bn1 = BatchNormalization(axis=-1)(dr1)
           
        forward2 = Conv1D(256,11,padding="same",activation=activation_C,name = 'conv2')(bn1)
        dr2 = Dropout(dropout_size)(forward2)
        bn2 = BatchNormalization(axis=-1)(dr2)
         
        pool2 = MaxPooling1D(3,strides = 2, padding='same')(bn2)
          
        forward3 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv3')(pool2)
        dr3 = Dropout(dropout_size)(forward3)
        bn3 = BatchNormalization(axis=-1)(dr3) 
        
        pool3 = MaxPooling1D(3,strides = 2,padding='same')(bn3)
          
        forward4 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv4')(pool3)
        dr4 = Dropout(dropout_size)(forward4)
        bn4 = BatchNormalization(axis=-1)(dr4) 
        pool4 = MaxPooling1D(3,strides = 2,padding='same')(bn4)
           
        forward5 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv5')(pool4)
        dr5 = Dropout(dropout_size)(forward5)
        bn5 = BatchNormalization(axis=-1,name='audio_branch')(dr5) 
        
        out_audio_channel = bn5 
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        
        return audio_sequence , out_audio_channel , audio_model
    
        
    def build_visual_model (self):
        
        [Xshape, Yshape] = self.input_dim
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape)
        visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
        
        forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        resh1 = Reshape([196,512],name='reshape_visual')(bn_visual) 
        
        out_visual_channel = resh1
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        return visual_sequence , out_visual_channel , visual_model
    
   
       
    def CNN0 (self):
    
        audio_sequence , out_audio_channel , audio_model = self.build_audio_model ()
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model ()  
        
        # post-processing for Audio and Image channels (Dense + L2 norm layers)
        
        dnA = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
        normA = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(dnA)
        
        dnI = Dense(512,activation='linear',name='dense_visual')(out_visual_channel) 
        normI = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(dnI)
      
        # combining audio and visual channels             
        A = normA
        I = normI
        
        mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_matchmap') 
        
        def final_layer(tensor):
            x= tensor 
            if self.model_subname == 'sisa':
                score = K.mean( (K.mean(x, axis=1)), axis=-1)
            elif self.model_subname == 'misa':
                score = K.mean( (K.mean(x, axis=1)), axis=-1)
            elif self.model_subname == 'sima':
                score = K.mean( (K.mean(x, axis=1)), axis=-1)         
            output_score = Reshape([1],name='reshape_final')(score)          
            return output_score
        
        lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)        
        final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = lambda_layer )
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= I, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence,outputs= A, name='visual_embedding_model')

        return final_model, visual_embedding_model, audio_embedding_model        
          

    def build_model (self):       
        final_model, visual_embedding_model, audio_embedding_model = self.CNN0()
        return final_model, visual_embedding_model, audio_embedding_model     


