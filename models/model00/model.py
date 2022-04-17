
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import  MaxPooling1D,  Conv1D,Conv2D, ReLU, Add
from tensorflow.keras.layers import Softmax, Permute, AveragePooling1D, Concatenate, dot

class VGS:
    
    def __init__(self):
        pass

    def build_audio_model (self, input_dim): 
        [Xshape, Yshape] = self.input_dim
        speech_sequence = Input(shape=Xshape) #Xshape = (512, 40)
  
        # speech channel
        
        #speech_sequence_masked = Masking (mask_value=0., input_shape=X2shape)(speech_sequence)
        strd = 2
        
        x0 = Conv1D(128,1,strides = 1, padding="same")(speech_sequence)
        x0 = BatchNormalization(axis=-1)(x0)
        x0 = ReLU()(x0) 
          
        # layer 1  
        in_residual = x0  
        x1 = Conv1D(128,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(128,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(128,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_1 = ReLU()(out) # (500, 128) 
        
        # layer 2
        in_residual = out_1  
        x1 = Conv1D(256,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(256,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(256,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_2 = ReLU()(out) # (256, 256)
        
        # layer 3
        in_residual = out_2  
        x1 = Conv1D(512,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(512,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(512,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_3 = ReLU()(out) # (128, 512)
    
        # layer 4
        # in_residual = out_3  
        # x1 = Conv1D(1024,9,strides = 1, padding="same")(in_residual)
        # x1 = BatchNormalization(axis=-1)(x1)
        # x1 = ReLU()(x1)       
        # x2 = Conv1D(1024,9,strides = strd, padding="same")(x1)  
        # downsample = Conv1D(1024,9,strides = strd, padding="same")(in_residual)
        # out = Add()([downsample,x2])
        # out_4 = ReLU()(out)   # (64, 1024)  
        
        # pooling
        
        # out_speech_channel  = AveragePooling1D(512,padding='same')(out_3) #(N,1, 1024)
        # out_speech_channel = Reshape([out_speech_channel.shape[2]])(out_speech_channel)  #(N, 1024)      
        # out_speech_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_speech')(out_speech_channel)
        out_speech_channel = out_3               
        audio_model = Model(inputs= speech_sequence, outputs = out_speech_channel )
        #audio_model.summary()
        return speech_sequence , out_speech_channel , audio_model 
    
          
    def build_simple_audio_model (self , input_dim):
        
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
        bn5 = BatchNormalization(axis=-1,name='audio_branch')(dr5) # (N,64,512)
        
        out_audio_channel = bn5 
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        
        return audio_sequence , out_audio_channel , audio_model
    
        
    def build_visual_model (self, input_dim):
        # Yshape = (14, 14, 512)
        [Xshape, Yshape] = self.input_dim
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape)
        
        visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
        
        forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        resh1 = Reshape([196,512],name='reshape_visual')(bn_visual) 
        
        out_visual_channel = resh1 # (N, 196,512)
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        return visual_sequence , out_visual_channel , visual_model
   

    def CNNatt (self , model_subname , input_dim):
        #input_dim = [(512, 40), (14, 14, 512)]
        speech_sequence , out_speech_channel , audio_model = self.build_audio_model (input_dim)
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model ( input_dim)          
        
        A = out_speech_channel
        I = out_visual_channel
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs = I, name='visual_embedding_model')
        audio_embedding_model = Model(inputs= speech_sequence, outputs = A, name='audio_embedding_model')  
        
        #### Attention I for query Audio
        # checks which part of image gets more attention based on audio query.   
        keyImage = out_visual_channel
        valueImage = out_visual_channel
        queryAudio = out_speech_channel
        
        scoreI = keras.layers.dot([queryAudio,keyImage], normalize=False, axes=-1,name='scoreI')
        weightID = Dense(196,activation='sigmoid')(scoreI)
        weightI = Softmax(name='weigthI')(scoreI)
        
        valueImage = Permute((2,1))(valueImage)
        attentionID = keras.layers.dot([weightID, valueImage], normalize=False, axes=-1,name='attentionID')
        attentionI = keras.layers.dot([weightI, valueImage], normalize=False, axes=-1,name='attentionI')
        
        poolAttID = AveragePooling1D(512, padding='same')(attentionID)
        poolAttI = AveragePooling1D(512, padding='same')(attentionI)
        
        poolqueryAudio = AveragePooling1D(512, padding='same')(queryAudio)
        
        outAttAudio = Concatenate(axis=-1)([poolAttI,poolAttID, poolqueryAudio])
        outAttAudio = Reshape([1536],name='reshape_out_attAudio')(outAttAudio)
    
        ###  Attention A  for query Image
        # checks which part of audio gets more attention based on image query.
        keyAudio = out_speech_channel
        valueAudio = out_speech_channel
        queryImage = out_visual_channel
        
        scoreA = keras.layers.dot([queryImage,keyAudio], normalize=False, axes=-1,name='scoreA')
        weightAD = Dense(64,activation='sigmoid')(scoreA)
        weightA = Softmax(name='weigthA')(scoreA)
        
        valueAudio = Permute((2,1))(valueAudio)
        attentionAD = keras.layers.dot([weightAD, valueAudio], normalize=False, axes=-1,name='attentionAD')
        attentionA = keras.layers.dot([weightA, valueAudio], normalize=False, axes=-1,name='attentionA')
        
        poolAttAD = AveragePooling1D(512, padding='same')(attentionAD)
        poolAttA = AveragePooling1D(512, padding='same')(attentionA)
        
        poolqueryImage = AveragePooling1D(512, padding='same')(queryImage)
        
        outAttImage = Concatenate(axis=-1)([poolAttA,poolAttAD, poolqueryImage])
        outAttImage = Reshape([1536],name='reshape_out_attImage') (outAttImage)
        
        # combining audio and visual channels 
        
        A_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(outAttAudio)
        I_e = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(outAttImage)
        
        
        if self.loss == "Triplet":
            mapIA = keras.layers.dot([I_e,A_e],axes=-1,normalize = True,name='dot_final')
            final_model = Model(inputs=[visual_sequence, speech_sequence], outputs = mapIA , name='vgs_model')
            
        elif self.loss == "MMS":
            s_output = Concatenate(axis=1)([Reshape([1 , I_e.shape[1]])(I_e) ,  Reshape([1 ,A_e.shape[1]])(A_e)])
            final_model = Model(inputs=[visual_sequence,  speech_sequence], outputs = s_output )
    
        return final_model, visual_embedding_model, audio_embedding_model 
         
    def CNN0 (self, model_subname, input_dim):
    
        #input_dim = [(512, 40), (14, 14, 512)]
        speech_sequence , out_speech_channel , audio_model = self.build_audio_model ( input_dim)
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model ( input_dim)  
        
        
        I = AveragePooling1D(512,padding='same') (out_visual_channel)
        I = Reshape([out_speech_channel.shape[2]])(I) # (N, 512)
        
        A = AveragePooling1D(512,padding='same') (out_speech_channel)
        A = Reshape([out_speech_channel.shape[2]])(A) # (N, 512)
         
        A_e = Dense(512,activation='linear',name='dense_audio')(A)       
        #A = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(A)
        
        I_e = Dense(512,activation='linear',name='dense_visual')(I) 
        #I = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(I)
        
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs = I_e, name='visual_embedding_model')
        audio_embedding_model = Model(inputs= speech_sequence, outputs = A_e, name='audio_embedding_model')  
        
        if self.loss == "Triplet":
            # mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_matchmap') 
            # def final_layer(tensor):
            #     x= tensor 
            #     score = K.mean( (K.mean(x, axis=1)), axis=-1)
            #     output_score = Reshape([1],name='reshape_final')(score)          
            #     return output_score
            # lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)
            # final_model = Model(inputs=[visual_sequence, speech_sequence], outputs = lambda_layer )
            
            mapIA = dot([I_e,A_e],axes=-1,normalize = True,name='dot_matchmap')       
            final_model = Model(inputs=[visual_sequence, speech_sequence], outputs = mapIA )
            
        elif self.loss == "MMS":
            s_output = Concatenate(axis=1)([Reshape([1 , I_e.shape[1]])(I_e) ,  Reshape([1 ,A_e.shape[1]])(A_e)])
            final_model = Model(inputs=[visual_sequence,  speech_sequence], outputs = s_output )
        return final_model, visual_embedding_model, audio_embedding_model
  
        return  final_model, visual_embedding_model, audio_embedding_model   
          
    def build_model (self, model_name, model_subname, input_dim): 
            
        if model_name == 'CNN0':
            final_model, visual_embedding_model, audio_embedding_model = self.CNN0(model_subname , input_dim)
        elif model_name == 'CNNatt':
            final_model, visual_embedding_model, audio_embedding_model = self.CNNatt(model_subname , input_dim)
    
        return final_model, visual_embedding_model, audio_embedding_model         
    

    
     


