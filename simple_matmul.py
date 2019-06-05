import tensorflow as tf
import tf_encrypted as tfe

def provide_input():
    return tf.ones(shape=(5,10))

w = tfe.define_private_variable(tf.ones(shape=(10,10)))
x = tfe.define_private_input('input-provider', provide_input)

y = tfe.matmul(x, w)

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer())
    result = sess.run(y.reveal())
    print(result)