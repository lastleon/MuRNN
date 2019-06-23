import tensorflow as tf
import music21 as mu

#### data params

# batch size is one, because inside a batch the number of timesteps must be the same
batch_size = 1
# timesteps have variable size
timesteps = None
# dimension of data
features = 999 #don't know yet


#### model params

epochs = 10
# ...

############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU