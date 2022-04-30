import tensorflow as tf

def add_5_times_5(x):
    return x + 5, x*5

def add_5_times_5_v2(x, y):
    return x +5, y * 5

if __name__=='__main__':
    dataset=tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6]).batch(3)
    dataset2=tf.data.Dataset.from_tensor_slices([7,6,5,4,3,21,9,4]).batch(2)

    dataset = tf.data.Dataset.zip((dataset,dataset2))
    # dataset=dataset.map(add_5_times_5)
    # dataset=dataset.map(add_5_times_5_v2)
    # dataset=dataset.shuffle(buffer_size=3)
    # dataset=dataset.batch(3)
    for a in dataset:
        print(a)
