###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)

 

###############################################################################

from train import train_validate



obj = train_validate ()
obj()


# to check some examples of similarity search
# Znames , check = obj.test_similarities()
# i = 15
# print(Znames[i])
# print(Znames[check['best_pairs'][i]])