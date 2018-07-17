"""
@author:Haien Zhang
@file:buildFeature_V1.PY
@time:2018/06/15
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from data_reader_rematch import get_all_batch_for_train

import os

from config import model_dir,one_hot_features,multi_hot_features,titles_0,Train_for_sub

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    alphas_dice = tf.get_variable('alpha_dice' + name, _x.get_shape()[-1],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
    input_shape = list(_x.get_shape())

    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)
    return alphas_dice * (1.0 - x_p) * _x + x_p * _x


def parametric_relu(_x, name=''):
    alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
class deepFM():
    def __init__(self,field_config,net_config):
        self.indices = {}
        self.values = {}
        self.shape = {}
        self.learning_rate = 0.001
        self.max_grads_norm = 10
        self.filed_config = field_config
        self.embedding_size = net_config['embedding_size']
        self.mode = net_config['mode']
        self.decay = net_config['decay']
        self.draw_graph()

    def print_activation(self,t):
        print(t.op.name, ' ', t.get_shape().as_list())
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def draw_graph(self):

        # print(self.filed_config["field_length"])
        self.graph = tf.Graph()
        with self.graph.as_default():

            def batch_normal(xs, n_out, ph_train):
                with tf.variable_scope('bn'):
                    batch_mean, batch_var = tf.nn.moments(xs, axes=[0])
                    beta = tf.Variable(tf.constant(0.0, shape=[n_out]))
                    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]))
                    epsilon = 1e-3

                    ema = tf.train.ExponentialMovingAverage(decay=0.5)

                    def mean_var_with_update():
                        ema_apply_op = ema.apply([batch_mean, batch_var])
                        with tf.control_dependencies([ema_apply_op]):
                            return tf.identity(batch_mean), tf.identity(batch_var)

                    mean, var = tf.cond(ph_train, mean_var_with_update,
                                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
                    xs_norm = tf.nn.batch_normalization(xs, mean, var, beta, gamma, epsilon)
                return xs_norm

            with tf.name_scope("sparse_placeholder") as scope:
                input_x = {}
                for item in one_hot_features:
                    # print(item)
                    self.indices[item] = tf.placeholder(shape=[None,2],dtype=tf.int64)
                    self.values[item] = tf.placeholder(shape=[None],dtype=tf.int64)
                    self.shape[item] = tf.placeholder(shape=[None],dtype=tf.int64)   #
                    input_x[item] = tf.SparseTensor(indices=self.indices[item],values=self.values[item],dense_shape=self.shape[item])
                for item in multi_hot_features:
                    self.indices[item] = tf.placeholder(shape=[None, 2], dtype=tf.int64)
                    self.values[item] = tf.placeholder(shape=[None], dtype=tf.int64)
                    self.shape[item] = tf.placeholder(shape=[None], dtype=tf.int64)  #
                    input_x[item] = tf.SparseTensor(indices=self.indices[item], values=self.values[item],
                                                  dense_shape=self.shape[item])
                self.label_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
                self.keep_prob = tf.placeholder(dtype=tf.float32)
                self.phase_train = tf.placeholder(dtype=tf.bool)
            self.see = input_x
            # print(input_x)
            with tf.name_scope("embedding_var") as scope:
                # embeddings = {}
                embeddings_w0 = {}
                embeddings_w1 = {}
                embeddings_w2 = {}
                embeddings_line = {}
                embeddings_mv = {}
                for item in one_hot_features:

                    embeddings_mv[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item], self.embedding_size],
                                                        minval=-0.01, maxval=0.01, seed=2018),
                        trainable=True, name="embedding_variables_{}".format(item))
                    embeddings_w0[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_w1[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_w2[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_line[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item], 1],
                                                        minval=0, seed=2018), trainable=True,
                        name="embedding_variables_{}".format(item))
                for item in multi_hot_features:

                    embeddings_w0[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_w1[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_w2[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item]], minval=1, maxval=1, seed=2018),
                        trainable=True, name="embedding_variables_w_{}".format(item))
                    embeddings_mv[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item], self.embedding_size],
                                                        minval=-0.01, maxval=0.01, seed=2018),
                        trainable=True, name="embedding_variables_{}".format(item))
                    embeddings_line[item] = tf.Variable(
                        initial_value=tf.random_uniform(shape=[self.filed_config[item], 1],
                                                        minval=0, seed=2018), trainable=True,
                        name="embedding_linear_variables_{}".format(item))

            for key in embeddings_mv:
                print(embeddings_mv[key])
            # y = {}
            w0, y_linear,y_mv,w1,w2= {},{},{},{},{}

            with tf.name_scope("embedding_lookup") as scope:

                # process one hot feature
                for item in one_hot_features:
                    w0[item] = tf.nn.embedding_lookup(embeddings_w0[item], self.values[item])
                    w0[item] = tf.SparseTensor(indices=self.indices[item], values=w0[item],
                                               dense_shape=self.shape[item])
                    w1[item] = tf.nn.embedding_lookup(embeddings_w1[item], self.values[item])
                    w1[item] = tf.SparseTensor(indices=self.indices[item], values=w1[item],
                                               dense_shape=self.shape[item])
                    w2[item] = tf.nn.embedding_lookup(embeddings_w2[item], self.values[item])
                    w2[item] = tf.SparseTensor(indices=self.indices[item], values=w2[item],
                                               dense_shape=self.shape[item])

                    y_mv[item] = tf.nn.embedding_lookup_sparse(embeddings_mv[item], input_x[item], sp_weights=w1[item],
                                                               combiner="mean")
                    y_linear[item] = tf.nn.embedding_lookup_sparse(embeddings_line[item], input_x[item],
                                                                   sp_weights=w2[item],
                                                                   combiner="mean")

                # process dict feature
                for item in multi_hot_features:
                    w0[item] = tf.nn.embedding_lookup(embeddings_w0[item], self.values[item])
                    w0[item] = tf.SparseTensor(indices=self.indices[item], values=w0[item],
                                               dense_shape=self.shape[item])
                    w1[item] = tf.nn.embedding_lookup(embeddings_w1[item], self.values[item])
                    w1[item] = tf.SparseTensor(indices=self.indices[item], values=w1[item],
                                               dense_shape=self.shape[item])
                    w2[item] = tf.nn.embedding_lookup(embeddings_w2[item], self.values[item])
                    w2[item] = tf.SparseTensor(indices=self.indices[item], values=w2[item],
                                               dense_shape=self.shape[item])
                    y_mv[item] = tf.nn.embedding_lookup_sparse(embeddings_mv[item], input_x[item], sp_weights=w1[item],
                                                               combiner="sum")
                    y_linear[item] = tf.nn.embedding_lookup_sparse(embeddings_line[item], input_x[item],
                                                                   sp_weights=w2[item],
                                                                   combiner="sum")

            if self.mode == "single":
                pass
            elif self.mode == "mixture":
                with tf.name_scope("concat") as scope:
                    source = []
                    source_line = [];source_mv=[]
                    for item in titles_0:
                        # source.append(y[item])
                        source_line.append(y_linear[item])
                        source_mv.append(y_mv[item])


                    L_y = len(source_mv)
                    # y = tf.concat(source, axis=1)
                    y_mv = tf.concat(source_mv, axis=1)
                    #------first order--------------------
                    y_linear = tf.concat(source_line, axis=1, name='y_linear')
                    # y_linear = tf.layers.dense(y_linear, 1, activation=None, use_bias=True)
                    # y_linear = batch_normal(y_linear, 1, ph_train=self.phase_train)
                    # self.print_activation(y_linear)

                    # y = tf.reshape(y, [-1, L_y, self.embedding_size])
                    y_mv = tf.reshape(y_mv, [-1, L_y, self.embedding_size])

                    # ----------two order part--------------------------
                    # y_sum_square = tf.pow(tf.reduce_sum(y, axis=1),2)    #None*k
                    # y_square_sum = tf.reduce_sum(tf.pow(y, 2), 1) #None*K
                    # y_two_order = 0.5*tf.subtract(y_sum_square, y_square_sum)
                    # y_two_order = self.batch_norm_layer(y_two_order, train_phase=self.phase_train, scope_bn='y_two_order')
                    # self.print_activation(y_two_order)

                    # ------------Multi view machine--------------
                    mvm_func = y_mv[:, 0, :]
                    for i in range(1, L_y):
                        mvm_func = tf.multiply(mvm_func, y_mv[:, i, :])
                    self.print_activation(mvm_func)
                    mvm_func = self.batch_norm_layer(mvm_func, train_phase=self.phase_train, scope_bn='mvm_func')
                    mvm_func = tf.reshape(mvm_func, shape=[-1, self.embedding_size])

                # y = tf.concat([y, y_mv], axis=2)
                y_mv = tf.transpose(y_mv, perm=[0, 2, 1])
                print(y_mv)
                """ Convolution neural network process """

                conv3 = tf.layers.dense(y_mv, 30, activation=None, use_bias=True)
                conv3 = batch_normal(conv3, 30, ph_train=self.phase_train)
                conv3 = dice(conv3, name='conv3')

                flatten = tf.reshape(conv3, [-1,  self.embedding_size*30])
                # length = flatten.shape[1].value
                y_concat = tf.concat([flatten,  mvm_func], axis=1)
                weight_size = y_concat.shape[1].value
                with tf.name_scope('fc') as scope:
                    num_neuron = 400
                    weight1 = tf.Variable(tf.random_uniform([weight_size, num_neuron], -0.01, 0.01,seed=2018), trainable=True)
                    # weight1 = tf.get_variable('weight1',shape=[length, num_neuron], initializer=tf.glorot_uniform_initializer(seed=1))
                    bias1 = tf.Variable(tf.constant(0.01, shape=[num_neuron]))
                    fc=tf.matmul(y_concat, weight1) + bias1
                    fc = batch_normal(xs=fc, n_out=num_neuron, ph_train=self.phase_train)
                    # fc = tf.nn.relu(fc)
                    fc = dice(fc,name='fc_1')
                    # fc = tf.nn.dropout(x=fc,keep_prob=self.keep_prob)

                    weight2 = tf.Variable(tf.random_uniform([num_neuron, num_neuron], -0.01, 0.01,seed=2018), trainable=True)
                    # weight2 = tf.get_variable('weight2', shape=[num_neuron, num_neuron], initializer=tf.glorot_uniform_initializer(seed=1))
                    bias2 = tf.Variable(tf.constant(0.001, shape=[num_neuron]))
                    fc = tf.matmul(fc, weight2) + bias2
                    fc = batch_normal(xs=fc, n_out=num_neuron, ph_train=self.phase_train)
                    # fc = tf.nn.relu(fc)
                    fc = dice(fc, name='fc_2')
                    # fc = tf.nn.dropout(x=fc, keep_prob=self.keep_prob)

                    weight3 = tf.Variable(tf.random_uniform([num_neuron, num_neuron], -0.01, 0.01,seed=2018), trainable=True)
                    # weight3 = tf.get_variable('weight3', shape=[num_neuron, num_neuron], initializer=tf.glorot_uniform_initializer(seed=1))
                    bias3 = tf.Variable(tf.constant(0.001, shape=[num_neuron]))
                    fc = tf.matmul(fc, weight3) + bias3
                    fc = batch_normal(xs=fc, n_out=num_neuron, ph_train=self.phase_train)
                    # fc = tf.nn.relu(fc)
                    fc = dice(fc, name='fc_3')
                    fc = tf.nn.dropout(x=fc, keep_prob=self.keep_prob)

                    # concat linear and dnn
                    fc = tf.concat([y_linear, fc], axis=1)
                    weight_final = fc.shape[1].value

                with tf.name_scope('output') as scope:
                    weight_o = tf.Variable(tf.random_uniform([weight_final, 1], -0.01, 0.01,seed=2018), trainable=True)
                    # weight_o = tf.get_variable('weight_o', shape=[num_ssneuron, 1], initializer=tf.glorot_uniform_initializer(seed=1),trainable=True)
                    bias1 = tf.Variable(tf.constant(0.001, shape=[1]))
                    logits = tf.matmul(fc, weight_o) + bias1
                    self.output = tf.nn.sigmoid(logits)

                self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=logits,multi_class_labels=self.label_holder))

                global_step = tf.Variable(0, trainable=False)

                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)

            with tf.name_scope("initial") as scope:
                self.init_op = tf.global_variables_initializer()


            self.saver = tf.train.Saver()

    def train(self,epoch = 2,):

        with tf.Session(graph=self.graph,) as sess:
            sess.run(self.init_op)
            train_loss_save = []
            valid_loss_save = []
            valid_auc_save = []
            pre_valid_loss = 10
            for i in range(epoch):
                print("echo:{}/{}".format(i+1,epoch))

                if Train_for_sub:
                    generator = get_all_batch_for_train(batchsize=548)
                    pass
                else:
                    pass
                    # generator = get_batch_for_train(batchsize=724)
                for j,batch_dict in enumerate(generator):

                    # print('\r{}'.format(j%1662),end=' ')
                    feed_dict = {}
                    for item in one_hot_features:
                        feed_dict[self.indices[item]] = batch_dict[item][0]
                        feed_dict[self.values[item]] = batch_dict[item][1]
                        feed_dict[self.shape[item]] = batch_dict[item][2]

                    for item in multi_hot_features:
                        feed_dict[self.indices[item]] = batch_dict[item][0]
                        feed_dict[self.values[item]] = batch_dict[item][1]
                        feed_dict[self.shape[item]] = batch_dict[item][2]
                    feed_dict[self.label_holder] = batch_dict['label']
                    feed_dict[self.keep_prob] = 0.3                 # 设置训练时的dropout
                    feed_dict[self.phase_train] = True
                    ops = [self.train_step,self.loss,]
                    _, loss= sess.run(ops, feed_dict=feed_dict)

                    # if not Train_for_sub:
                    #     if (j + 1) % 1258 == 0:
                    #         print('\n')
                    #         generator_valid_set = build_valid_set(553)
                    #         loss_valid = []
                    #         pre_pro_list = []
                    #         valid_labels = []
                    #         for single_valid_dict in generator_valid_set:
                    #             feed_dict_valid = {}
                    #             for item in one_hot_features:
                    #                 feed_dict_valid[self.indices[item]] = single_valid_dict[item][0]
                    #                 feed_dict_valid[self.values[item]] = single_valid_dict[item][1]
                    #                 feed_dict_valid[self.shape[item]] = single_valid_dict[item][2]
                    #
                    #             for item in multi_hot_features:
                    #                 feed_dict_valid[self.indices[item]] = single_valid_dict[item][0]
                    #                 feed_dict_valid[self.values[item]] = single_valid_dict[item][1]
                    #                 feed_dict_valid[self.shape[item]] = single_valid_dict[item][2]
                    #             feed_dict_valid[self.label_holder] = single_valid_dict['label']
                    #             feed_dict_valid[self.keep_prob] = 1.0
                    #             feed_dict_valid[self.phase_train] = False
                    #             single_valid_label = single_valid_dict['label']
                    #             ops_valid =  [self.loss,self.output]
                    #             loss_tmp,pre_pro = sess.run(ops_valid,feed_dict=feed_dict_valid)
                    #
                    #             loss_valid.append(loss_tmp)
                    #             pre_pro_list.extend(pre_pro.reshape([-1]))
                    #             valid_labels.extend(single_valid_label.reshape([-1]))

                            # auc_value = roc_auc_score(valid_labels, pre_pro_list)
                            # loss_valid = sum(loss_valid)/len(loss_valid)
                            # train_loss_save.append(loss)
                            # valid_loss_save.append(loss_valid)
                            # valid_auc_save.append(auc_value)
                            # print('*' * 50, "modify", "*" * 40)
                            # print("iter:",j+1,"train set batch-loss:",loss,"  valid set loss:",loss_valid,"valid auc:",auc_value)
                            # print('*' * 50, "modify", "*" * 40)
                            # if loss_valid < pre_valid_loss:
                            #     model_dir = "./deepfm_model_tune_hai1/tfmodel.ckpt"
                            #     self.saver.save(sess, model_dir)
                            #     print("model updated!")
                            #     pre_valid_loss = loss_valid
                            # print('*' * 50, "modify", "*" * 40)
                            # print('\n')
                if Train_for_sub:
                    self.saver.save(sess, model_dir)
                    print("model updated!")


    def predict(self,test_data_gen):

        with tf.Session(graph=self.graph) as sess:
            print('restoring ......')
            self.saver.restore(sess=sess,save_path=model_dir)
            print('restore ok!')
            # self.saver.recover_last_checkpoints(checkpoint_paths="/home/tensor/Desktop/tencent/contest_new5_9/functions/tfmodel/deepfm_model_hai/checkpoint")
            pre_pro_list = []
            cnt = 0
            for dict_sub_test in test_data_gen:
                # print(dict_sub_test)
                feed_dict_test = {}
                for item in one_hot_features:
                    # print(item)
                    feed_dict_test[self.indices[item]] = dict_sub_test[item][0]
                    feed_dict_test[self.values[item]] = dict_sub_test[item][1]
                    feed_dict_test[self.shape[item]] = dict_sub_test[item][2]

                for item in multi_hot_features:
                    feed_dict_test[self.indices[item]] = dict_sub_test[item][0]
                    feed_dict_test[self.values[item]] = dict_sub_test[item][1]
                    feed_dict_test[self.shape[item]] = dict_sub_test[item][2]
                # feed_dict_test[self.label_holder] = dict_sub_test['label']
                feed_dict_test[self.keep_prob] = 1.0
                feed_dict_test[self.phase_train] = False
                pre_pro = sess.run(self.output,feed_dict=feed_dict_test)
                pre_pro_list.extend(pre_pro.reshape([-1]))
                cnt +=1
                print("\r {}".format(cnt),end=" ")
            return pre_pro_list





