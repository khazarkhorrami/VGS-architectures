
import numpy
import scipy.spatial as ss

def calculate_recallat10( embedding_1,embedding_2, sampling_times, number_of_all_audios, pool):   
    recall_all = []
    recallat = 10  
    for trial_number in range(sampling_times):      
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=pool)       
        vec_1 = [embedding_1[item] for item in data_ind]
        vec_2 = [embedding_2[item] for item in data_ind]           
        distance_utterance = ss.distance.cdist( vec_1 , vec_2 ,  'cosine') # 1-cosine
       
        r = 0
        for n in range(pool):
            ind_1 = n #random.randrange(0,number_of_audios)                   
            distance_utterance_n = distance_utterance[n]            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_1)*1)   
        recall_all.append(r)
        del distance_utterance  
        
    return recall_all
