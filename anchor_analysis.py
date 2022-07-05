import tensorflow as tf


# with tf.compat.v1.variable_scope('share', reuse=tf.compat.v1.AUTO_REUSE):
#     # global_epochs = tf.compat.v1.get_variable("global_epochs", shape=[], trainable=False,  )
#     global_epochs = tf.Variable(1, trainable=False, name="global_epochs")
# print('global_epochs', global_epochs)

# with tf.compat.v1.variable_scope('share', reuse=tf.compat.v1.AUTO_REUSE):
#     v1 = tf.compat.v1.get_variable("global_epochs", shape=[], trainable=False, reuse=True)
# print('v1', v1)

# print(global_epochs is v1)

#########################################################################
# @tf.function
# def foo():
#     with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
#         v = tf.compat.v1.get_variable(name="v", shape=[], initializer=lambda shape,dtype: 1)
#         return v

# @tf.function
# def assign_add(v):
#   v.assign_add(1)
#   return v

# with tf.compat.v1.variable_scope("foo", reuse=tf.compat.v1.AUTO_REUSE):
#     vv= tf.Variable(111, trainable=False, dtype=tf.int32)
#     tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, vv)
#     vv.assign_add(1)

#     v = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

# print(v)
# v[0].assign_add(1)
# print(v[0])

# v1 = foo()  # Creates v.
# v2 = foo()  # Gets the same, existing v.
# print(v1)
# print(v2)
# assert v1 == v2



# with tf.Graph().as_default() as g:
#     g.get_tensor_by_name("foo/v")
############################################
# a = tf.constant(1.3, name='const_a')
# b = tf.Variable(3.1, name='variable_b')
# c = tf.add(a, b, name='addition')
# d = tf.multiply(c, a, name='multiply')

# current_graph = tf.compat.v1.get_default_graph()
# all_names = [op.name for op in current_graph.get_operations()]

# print(all_names)

#####################################################################
def init_shared_variable():
            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64, name="global_steps")  
            global_epochs = tf.Variable(1, trainable=False, dtype=tf.int64, name="global_epochs")
            tf.compat.v1.add_to_collection("shared", global_steps) 
            tf.compat.v1.add_to_collection("shared", global_epochs) 
            print(tf.compat.v1.get_collection("shared"))

@tf.function
def get_shared_variable():
    print(tf.compat.v1.get_collection("shared"))
@tf.function
def test():
    a,b = get_shared_variable()
    tf.print(a,b)

g = init_shared_variable()
g2 = get_shared_variable()
print(g, g2)
print(g is g2)
exit()
c = a
a.assign_add(1)
print(a, c)
test()