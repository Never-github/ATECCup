from tensorflow.train import list_variables, load_variable
import tensorflow.train
import tensorflow as tf
now_ckpt = '/Users/ziangcui/Desktop/ATECCup/bert_cls/new_ckpt/mymodel.ckpt'
with tf.Session() as sess:
    va = list_variables(now_ckpt)
    var_list = []
    for name, j in va:
        if name=='bert/embeddings/word_embeddings':
            var = load_variable(now_ckpt, name)
            var_list.append(tf.Variable(var, name=name))
            print(type(var))
        else:
            var = load_variable(now_ckpt, name)
            var_list.append(tf.Variable(var, name=name))

    saver = tf.train.Saver(var_list=var_list)


    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'bert_cls/newnewckpt/mymodel.ckpt')
