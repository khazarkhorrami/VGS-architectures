
"""

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot, Softmax, Permute, UpSampling1D, Masking
from tensorflow.keras.layers import  MaxPooling1D,AveragePooling1D,  Conv1D, Concatenate, ReLU, Add, Multiply, GRU
from tensorflow.keras.optimizers import Adam, SGD

def build_apc ( Xshape):      
    audio_sequence = Input(shape=Xshape) #Xshape = (995, 40)
    prenet =  Conv1D(128,  kernel_size=1, name = 'prenet', activation='relu')(audio_sequence)
    prenet = BatchNormalization(axis=-1)(prenet)
    #prenet = Conv1D(128,3, activation='relu', padding='causal')(prenet1)      
    #context = GRU(256, return_sequences=True, name = 'GRU')(prenet3) # (N, 1000, 256)
    context = Conv1D(128, kernel_size=5, padding='causal', dilation_rate=1, activation='relu' , name = 'context1')(prenet)
    context = Conv1D(128, kernel_size=5, padding='causal', dilation_rate=2, activation='relu', name = 'context2')(context)
    context = Conv1D(128, kernel_size=5, padding='causal', dilation_rate=4, activation='relu', name = 'context3')(context)
  
    postnet_audio =  Conv1D(40,  kernel_size=1, name = 'postnet')(context)

    predictor = Model(audio_sequence, postnet_audio)
    predictor.compile(optimizer=SGD(lr=1e-03), loss='mean_absolute_error')
    apc = Model(audio_sequence, context)
   
    return predictor, apc


def produce_apc_features (data, outputdir):
    predictor, apc = build_apc(Xshape= (512, 40)) 
    predictor.summary()
    apc.summary()
    predictor.load_weights('%smodel_weights.h5' % outputdir)
    apc.layers[1].set_weights = predictor.layers[1].get_weights
    apc.layers[2].set_weights = predictor.layers[2].get_weights
    apc.layers[3].set_weights = predictor.layers[3].get_weights
    apc.layers[4].set_weights = predictor.layers[4].get_weights
    apc_features = apc.predict(data)
    return apc_features

def train_apc (audio_features_train, audio_features_val):
    l = 5
    predictor, apc = build_apc(Xshape= (512, 40))         
    val_loss_init = 1000
    predictor.summary()

    xtrain = audio_features_train[:,0:-l,:]
    ytrain = audio_features_train[:,l:,:]        
    
    xval = audio_features_val[:,0:-l,:]
    yval = audio_features_val[:,l:,:]      

    for epoch in range(50):
        predictor.fit(xtrain, ytrain , epochs=5, shuffle = True, batch_size=120)
        val_loss = predictor.evaluate(xval, yval, batch_size=120)       
        if val_loss[0] < val_loss_init:
            val_loss_init = val_loss[0]
            weights = predictor.get_weights()
            predictor.set_weights(weights)
            #predictor.save_weights('%smodel_weights.h5' % outputdir)
            
# def build_apc_visual (self, Xshape):
#     #visual block
#     visual_sequence = Input(shape=(10,2048))  
#     prenet_visual = Dense(256, activation='relu', name = 'prenetvisual')(visual_sequence)
#     pool_visual = UpSampling1D(100, name = 'upsamplingvisual')(prenet_visual) # (N, 1000, 256)
#     #forward_visual = Dense(256,activation='relu')(pool_visual)#(N, 1000, 256)
#     out_visual = pool_visual[:,:-5,:]
    
    
#     audio_sequence = Input(shape=Xshape) #Xshape = (995, 40)
#     prenet = Dense(256, activation='relu', name = 'prenet')(audio_sequence)
#     #prenet = Conv1D(128,3, activation='relu', padding='causal')(prenet1)
    
#     #context = GRU(256, return_sequences=True, name = 'GRU')(prenet) # (N, 1000, 256)
#     context = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=1,name = 'context1')(prenet)
#     context = Conv1D(256, kernel_size=3, padding='causal', dilation_rate=2, name = 'context2')(context)
#     context = Conv1D(256, kernel_size=3, padding='causal', dilation_rate=4, name = 'context3')(context)
#     context_audiovisual = Add()([context, out_visual])# (N, 1000, 256)  
#     postnet_audiovisual = Conv1D(40, kernel_size=1, padding='same')(context_audiovisual) # (1000, 40) 
#     predictor = Model(audio_sequence,postnet_audiovisual)
#     predictor.compile(optimizer=Adam(lr=1e-04), loss='mean_absolute_error',  metrics=['accuracy'])
#     apc = Model(audio_sequence, context)   
#     return predictor, apc