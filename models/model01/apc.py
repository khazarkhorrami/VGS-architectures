
"""

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot, Softmax, Permute, UpSampling1D, Masking
from tensorflow.keras.layers import  MaxPooling1D,AveragePooling1D,  Conv1D, Concatenate, ReLU, Add, Multiply, GRU

# activation='relu'
def build_apc ( Xshape):      
    audio_sequence = Input(shape=Xshape) #Xshape = (995, 40)
    prenet =  Dense(512, name = 'prenet')(audio_sequence)
    #prenet = BatchNormalization(axis=-1)(prenet)

    # network 0 (short recieptive field (5 cells for kernel = 2)) # loss: -0.87
    # network 0 (short recieptive field (16 cells for kernel = 4)) # loss: 
    context0 = Conv1D(256, kernel_size=4, padding='causal')(prenet)
    context0 = Conv1D(256, kernel_size=4, padding='causal')(context0)
    context0 = Conv1D(256, kernel_size=4, padding='causal')(context0)
    context0 = Conv1D(256, kernel_size=4, padding='causal')(context0)
    
    # network 1 (long recieptive field: standard dilated cnn) # loss: -0.85
    # context1 = Conv1D(256, kernel_size=2, padding='causal', dilation_rate=1)(prenet)
    # context1 = Conv1D(256, kernel_size=2, padding='causal', dilation_rate=2)(context1)
    # context1 = Conv1D(256, kernel_size=2, padding='causal', dilation_rate=4)(context1)
    # context1 = Conv1D(256, kernel_size=2, padding='causal', dilation_rate=8)(context1)
    
    # residual layer
    #context = Add()([context0, context1])
    
    context = context0
    # network 2 (residualized)
    # context1 = Conv1D(128, kernel_size=4, padding='causal', dilation_rate=1)(prenet)
    # context2 = Conv1D(128, kernel_size=2, padding='causal', dilation_rate=2)(context1)
    # context_12 = Add()([context1,context2])
    # context3 = Conv1D(128, kernel_size=2, padding='causal', dilation_rate=4)(context_12)
    # context_123 = Add()([context_12,context3])
    # context4 = Conv1D(128, kernel_size=2, padding='causal', dilation_rate=8)(context_123)
    # context = Add()([context_123,context4])
    
    postnet_audio =  Conv1D(40, kernel_size=1 , name = 'postnet')(context)

    predictor = Model(audio_sequence, postnet_audio)
    
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
