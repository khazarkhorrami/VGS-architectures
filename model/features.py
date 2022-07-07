
"""
"""

import json
import os
import numpy
import pickle
from sklearn.utils import shuffle

  
    
def get_SPOKENCOCO_data (json_file , split):
        
    infile = open(json_file, 'rb')
    content = json.load(infile)
    infile.close()

    json_data = content['data']   
    data = []
    #iterating over images
    for json_item in json_data:
        item_dict = {}
        img_fullname = json_item['image']
        img_data = json_item['captions']
        
        item_dict ['image_name'] = img_fullname     
        img_feature_name = img_fullname.split('/')[1][:-4]     
        item_dict ['visual_feature'] = os.path.join(split,img_feature_name )
        
        item_dict['audio_data'] = []
        # iterating over captions of each image
        for caption_item in img_data:
            audio_info = {}
            
            wav_name = caption_item ['wav']
            audio_info ['wav_name'] = wav_name
            audio_info ['audio_feature'] = os.path.join(wav_name.split('/')[1],
                                                        wav_name.split('/')[2],
                                                        wav_name.split('/')[3][:-4])
                                                        
            audio_info ['text_caption'] = caption_item ['text']
            item_dict['audio_data'].append(audio_info)
        data.append(item_dict)
    return data 
    
def expand_feature_filenames (chunk):
    
    Ydata = []
    Xdata = []
    Zdata = []    
    #iterating over images
    for element in chunk:
        number_of_captions = len(element['af'])
        
        #iterating over captions per image
        for counter in range(number_of_captions):
            Ydata.append(element['vf'])
            Xdata.append(element['af'][counter])
            Zdata.append(element['captions'][counter])       
    return Ydata, Xdata , Zdata
        
def expand_feature_filenames2 (chunk, captionID):
    
    Ydata = []
    Xdata = []
    Zdata = []    
    #iterating over images
    for element in chunk:       
        # adding the specified captions       
        Ydata.append(element['vf'])
        Xdata.append(element['af'][captionID])
        Zdata.append(element['captions'][captionID])       
    return Ydata, Xdata , Zdata  

      
def read_feature_filenames (path_json, split, shuffle_data = True ):
    
    if split == 'train':
        json_name = 'SpokenCOCO_train.json'
    elif split == 'val':
        json_name = 'SpokenCOCO_val.json'
    json_file = os.path.join(path_json , json_name)
    
    data = get_SPOKENCOCO_data (json_file , split)
    feature_names = []
   
     
    #iterating over images
    for element in data:
        features = {}

        features['vf'] = element['visual_feature']
        features['af'] = []
        features['captions'] = []
        audio_data = element['audio_data']
        #iterating over captions per image
        for caption_item in audio_data:
            
            features['af'].append(caption_item['audio_feature'])
            features['captions'].append(caption_item['text_caption'])
            
        feature_names.append(features)
                
    if shuffle_data:
        inds_shuffled = shuffle(numpy.arange(len(feature_names)))       
    else:
        inds_shuffled = numpy.arange(len(feature_names))
        
    feature_names_output = [feature_names[ind] for ind in inds_shuffled]
        
    return feature_names_output  
    
    
def load_data (filepath):
    infile = open(filepath ,'rb')
    data = pickle.load(infile)
    infile.close()
    return data
    
def mvn (af_item):
    mean_item = numpy.mean(af_item)
    var_item = numpy.var(af_item)
    if var_item < 0.0000001:
        var_item = 1  
    af_norm = (af_item - mean_item ) /var_item
    return af_norm

def preparX (Xdata, len_of_longest_sequence, normalize):
    number_of_audios = numpy.shape(Xdata)[0]
    number_of_audio_features = numpy.shape(Xdata[0])[1]
    # normalizing  
    Xdata_norm = []
    for af_item in Xdata:
        #print(af_item.shape)
        af_normalized = mvn (af_item)
        Xdata_norm.append((af_normalized))
    if normalize:
        Xdata_final = Xdata_norm
    else:
        Xdata_final = Xdata
        
    # zero-padding
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       x_item = Xdata_final[k]
       x_item = x_item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(x_item):, :] = x_item
    return X

def prepareX_apc(Xnames, feature_path_audio, len_of_longest_sequence, normalize, overlap):
    Xdata = []
    for Xname in Xnames:
        Xpath = os.path.join(feature_path_audio, Xname )
        Xdata.append(load_data(Xpath))   
        
    #number_of_audios = numpy.shape(Xdata)[0]
    number_of_audio_features = numpy.shape(Xdata[0])[1]
    # normalizing  
    Xdata_norm = []
    for af_item in Xdata:
        af_normalized = mvn (af_item)
        Xdata_norm.append((af_normalized))
    if normalize:
        Xdata_final = Xdata_norm
    else:
        Xdata_final = Xdata
        
    # concatenating
    overlap_size = 100    
    X_concat = []
    for af_item in Xdata_final:
        X_concat.extend(af_item)
    X_concat = numpy.array(X_concat)
    number_of_samples = int (len(X_concat) / len_of_longest_sequence)
    Xdata_apc = numpy.zeros((number_of_samples,len_of_longest_sequence + overlap_size, number_of_audio_features),dtype ='float32')
    #adding the first item manually because of overlap > 0
    Xdata_apc[0, :,:] = X_concat[0:len_of_longest_sequence + overlap_size , :]
    for n in range(number_of_samples -1):
        Xdata_apc[n+1, :,:] = X_concat[ (( n +1 )*len_of_longest_sequence ) - overlap_size  : (n+2)*len_of_longest_sequence,:]
        
    X = numpy.array(Xdata_apc)
    return X

def preparY (dict_vgg):
    Y = numpy.array(dict_vgg)    
    return Y


def prepare_XY (Ynames, Xnames , feature_path_audio , feature_path_image , len_of_longest_sequence, normalize):
     #...................................................................... Y 
    Ydata = []
    for Yname in Ynames:
        Ypath = os.path.join(feature_path_image, Yname )
        Ydata_initial = load_data(Ypath)
        Ydata.append(numpy.reshape(Ydata_initial,[14,14,512]))
    #.................................................................. X
    Xdata = []
    for Xname in Xnames:
        Xpath = os.path.join(feature_path_audio, Xname )
        Xdata.append(load_data(Xpath))   
    # ......... preparing data (zeropadding for X) 
    Ydata_output = preparY(Ydata)
    Xdata_output = preparX(Xdata, len_of_longest_sequence, normalize)
    return Ydata_output, Xdata_output 
