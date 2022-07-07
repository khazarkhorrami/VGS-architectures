
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, Masking, Conv3D, MaxPooling3D
from tensorflow.keras.layers import  MaxPooling1D,  Conv1D,Conv2D, ReLU, Add
from tensorflow.keras.layers import Softmax, Permute, AveragePooling1D, Concatenate, dot


class CNN0:
    
    def __init__(self, input_dim):
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
      

    def build_model (self):       
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
            
            score = K.mean( (K.mean(x, axis=1)), axis=-1)
       
            output_score = Reshape([1],name='reshape_final')(score)          
            return output_score
        
        lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)        
        final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = lambda_layer )
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= I, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence,outputs= A, name='visual_embedding_model')
        
        return final_model, visual_embedding_model, audio_embedding_model     

     
class CNNatt:
    
    def __init__(self, input_dim):
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
    
    def build_model (self):
        #model_subname = self.model_subname
        audio_sequence , out_audio_channel , audio_model = self.build_audio_model ()
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model ()          
        
        A = out_audio_channel
        I = out_visual_channel
        
        #### Attention I for query Audio
        # checks which part of image gets more attention based on audio query.   
        keyImage = out_visual_channel
        valueImage = out_visual_channel
        queryAudio = out_audio_channel
        
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
        keyAudio = out_audio_channel
        valueAudio = out_audio_channel
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
        
        out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(outAttAudio)
        out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(outAttImage)
         
        
        mapIA = keras.layers.dot([out_visual,out_audio],axes=-1,normalize = True,name='dot_final')
        final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA , name='vgs_model')
        
         
        visual_embedding_model = Model(inputs=visual_sequence, outputs= I, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence,outputs= A, name='visual_embedding_model')

        return final_model, visual_embedding_model, audio_embedding_model



class AVLnet:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
             
          
    def build_audio_model (self): 
        
        # resDAVEnet
        
        [X1shape, X2shape, Yshape] = self.input_dim
        audio_sequence = Input(shape=X1shape) #X1shape = (21, 1024)
        speech_sequence = Input(shape=X2shape) #X2shape = (1000, 40)
        
        # audio channel
        # audio_sequence_masked = Masking (mask_value=0., input_shape=X1shape)(audio_sequence)
        # strd = 2
        
        # a0 = Conv1D(512,1,strides = 1, padding="same")(audio_sequence_masked)
        # a0 = BatchNormalization(axis=-1)(a0)
        # a0 = ReLU()(a0) #(21,1024)
        
        # out_sound_channel = UpSampling1D(3, name = 'upsampling_sound')(a0)
        # out_sound_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_sound')(out_sound_channel)
        
        # speech channel
        speech_sequence_masked = Masking(mask_value=0., input_shape=X2shape)(speech_sequence)
        strd = 2
        
        x0 = Conv1D(128,1,strides = 1, padding="same")(speech_sequence_masked)
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
        in_residual = out_3  
        x1 = Conv1D(1024,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(1024,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(1024,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_4 = ReLU()(out)   # (64, 1024)  
        
        out_speech_channel = out_4
        out_speech_channel  = AveragePooling1D(64,padding='same')(out_4) 
        out_speech_channel = Reshape([out_speech_channel.shape[2]])(out_speech_channel) 
        
          
        out_speech_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_speech')(out_speech_channel)
        
        # combining sound and speech branches
        #out_audio_channel = Concatenate(axis=-1)([out_sound_channel, out_speech_channel])
        out_audio_channel = out_speech_channel
        
        audio_model = Model(inputs= [audio_sequence, speech_sequence], outputs = out_audio_channel )
        audio_model.summary()
        return audio_sequence , speech_sequence  , out_audio_channel , audio_model   
    
        
    def build_visual_model (self):
        
        [X1shape, X2shape, Yshape] = self.input_dim
        
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape) #Yshape = (10,7,7,2048)
        visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
        
        forward_visual = Conv3D(1024,(1,3,3),strides=(1,1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        
        # 
        # visual_sequence_reshaped.shape
        
        #max pooling
        # pool_visual = MaxPooling2D((10,1),padding='same')(visual_sequence_reshaped)
        # out_visual_channel = Reshape([pool_visual.shape[2], pool_visual.shape[3]])(pool_visual)
        
        pool_visual = MaxPooling3D((10,7,7),padding='same')(bn_visual)
        input_reshape = pool_visual
        out_visual_channel = Reshape([input_reshape.shape[4]])(input_reshape)
        # out_visual_channel = Reshape([input_reshape.shape[2]*input_reshape.shape[3],
        #                                                          input_reshape.shape[4]], name='reshape_visual')(input_reshape)    
        out_visual_channel.shape
        out_visual_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1), name='lambda_visual')(out_visual_channel)
        
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

    def build_model(self, X1shape , X2shape , Yshape):
           
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)
        if self.audio_model_name == "resDAVEnet":        
            audio_sequence , speech_sequence  , out_audio_channel , audio_model   = self.build_resDAVEnet (X1shape , X2shape)            
        elif  self.audio_model_name == "simplenet": 
            audio_sequence , out_audio_channel , audio_model = self.build_simple_audio_model (X1shape , X2shape)
        
        
        V = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_visual-final') (out_visual_channel)
        A = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_audio-final') (out_audio_channel)
        
        # V = Reshape([1,V.shape[1], V.shape[2]])(V)
        # A = Reshape([1,A.shape[1], A.shape[2]])(A)
        # VA = dot ([V,A],axes=(-1,-1),normalize = True,name='batchdot')
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= V, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=[audio_sequence , speech_sequence], outputs= A, name='visual_embedding_model')
       
        
        # V_old = AveragePooling1D(64,padding='same')(V)
        # V_old = Reshape([V_old.shape[-1]])(V_old) 

        # A_old = AveragePooling1D(64,padding='same')(A)
        # A_old = Reshape([V_old.shape[-1]])(A_old)         


        # old = K.batch_dot(K.expand_dims(V_old,0), K.expand_dims(A_old,0) , axes=(-1,-1))  
        # old = K.squeeze(old, axis = 0)
        
       
 
        #new = K.squeeze(new, axis = 0)
    
        
        if self.loss == "triplet":  
            mapIA = dot([V,A],axes=-1,normalize = True,name='dot_matchmap')
            mapIA.shape
            # def final_layer(tensor):
            #     x= tensor 
            #     score = K.mean( (K.max(x, axis=1)), axis=-1)        
            #     output_score = Reshape([1],name='reshape_final')(score)          
            #     return output_score
            # lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)  
              
            final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence ]], outputs = mapIA)
           

            
        elif self.loss == "MMS":
            #s_output = Concatenate(axis=1)([Reshape([1 , V.shape[1], V.shape[2]])(V) ,  Reshape([1 ,A.shape[1], A.shape[2]])(A)])
            # def final_layer(input_tensor):
            #     return input_tensor
            
            # lambda_layer = Lambda(final_layer , name="final_layer" ) ([V,A])     #lambda x,y: [x,y]  
            
            VV = keras.backend.expand_dims(V,0)
            #AA = K.expand_dims(A,0)
            S = keras.backend.batch_dot(VV, A, axes =[-1,-1])
            def final_layer(tensor): #N,N,49,63
                x= tensor 
                score = K.mean( (K.max(x, axis=-1)), axis=-1)        
                     
                return score
            lambda_layer = Lambda(final_layer, name="final_layer")(S)
            final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence]], outputs = lambda_layer)
            final_model.summary()

        return final_model, visual_embedding_model,audio_embedding_model  


