import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data import *

global_step = tf.Variable(0, trainable=False, name='global_step')

# Hyperparameter setting
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.97, staircase=False)
K = 5
dropout_prob = 0.85
reg_coeff = 10**(-2)
batch_size = 64
epochs = 400
hidden_node = 10
initial_std = 0.02

rmse_avg = 0
mae_avg = 0

mae_best = 100
rmse_best = 100

saver_path = 'pretrained/'
model_name = 'best.ckpt'

test_file = 'test.txt'
result_file = 'result.txt'

##test file
with open(test_file, 'r') as file:
    lines = file.readlines()
    test_materials = []
    test_y_data = []
    sys = []
    ads_type = []
    for i in range(len(lines)):
        line = lines[i]
        str_list = line.strip().split(' ')
        test_materials.append(
            str_list[0] + '-' + str_list[1] + '-' + str_list[2] + '-' + str_list[3] + '-' + str_list[4] + '-' + str_list[5])
        test_y_data.append(float(str_list[6]))            
        sys.append(int(str_list[1]))
        ads_type.append(int(str_list[0]))
    test_materials = np.array(test_materials)
    test_y_data = np.array(test_y_data)


y = tf.placeholder(tf.float32, [None, 1], name='ads_energy')
keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

atoms_b = tf.placeholder(tf.float32, [None, None, CATEGORY_NUM], name='bulk_atoms')
atom_number_b = tf.placeholder(tf.float32, [None], name='atom_number_bulk')
atom_num_b = tf.placeholder(tf.int32, name='atom_num_bulk')
bond_number_b = tf.placeholder(tf.int32, name='bond_number_bulk')
bonds_b = tf.placeholder(tf.float32, [None, None, None, BOND_CATEGORY_NUM], name='bulk_bonds')
bonds_b_index1 = tf.placeholder(tf.int32, [None, None, None], name='bond_bulk_index1')
bonds_b_index2 = tf.placeholder(tf.int32, [None, None, None], name='bond_bulk_index2')

atoms_s = tf.placeholder(tf.float32, [None, None, CATEGORY_NUM], name='surface_atoms')
atom_number_s = tf.placeholder(tf.float32, [None], name='atom_number_surface')
atom_num_s = tf.placeholder(tf.int32, name='atom_num_surface')
bond_number_s = tf.placeholder(tf.int32, name='bond_number_surface')
bonds_s = tf.placeholder(tf.float32, [None, None, None, BOND_CATEGORY_NUM], name='surface_bonds')
bonds_s_index1 = tf.placeholder(tf.int32, [None, None, None], name='bond_surface_index1')
bonds_s_index2 = tf.placeholder(tf.int32, [None, None, None], name='bond_surface_index2')

## bulk part
atoms_dummy_bulk = tf.zeros([1, CATEGORY_NUM], dtype=tf.float32)
atoms_b_reshape = tf.reshape(atoms_b, [-1, CATEGORY_NUM])
atoms_b_reshape = tf.concat([atoms_b_reshape, atoms_dummy_bulk], axis=0)
bonds_b_reshape = tf.reshape(bonds_b, [-1, BOND_CATEGORY_NUM])
bonds_b_index1_reshape = tf.reshape(bonds_b_index1, [-1, ])
bonds_b_index2_reshape = tf.reshape(bonds_b_index2, [-1, ])

with tf.name_scope("bulk"):
    Wf_b_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution_filter_f')
    bf_b_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution_bias_f')
    Ws_b_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution_filter_s')
    bs_b_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution_bias_s')

    Wf_b_2 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution2_filter_f')
    bf_b_2 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution2_bias_f')
    Ws_b_2 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution2_filter_s')
    bs_b_2 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution2_bias_s')

    Wf_b_3 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution3_filter_f')
    bf_b_3 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution3_bias_f')
    Ws_b_3 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution3_filter_s')
    bs_b_3 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Bulk_convolution3_bias_s')

z_b_i_1 = tf.gather(atoms_b_reshape, bonds_b_index1_reshape)
z_b_j_1 = tf.gather(atoms_b_reshape, bonds_b_index2_reshape)
z_b_k_1 = tf.concat([z_b_i_1, z_b_j_1, bonds_b_reshape], axis=1)

## convolution layer1
sig_term_b = tf.sigmoid(tf.matmul(z_b_k_1, Wf_b_1) + tf.reshape(bf_b_1, [1, CATEGORY_NUM]))
relu_term_b = tf.nn.elu(tf.matmul(z_b_k_1, Ws_b_1) + tf.reshape(bs_b_1, [1, CATEGORY_NUM]))
conv_term_b = tf.multiply(sig_term_b, relu_term_b)
conv_term_b = tf.reshape(conv_term_b, [-1, atom_num_b, bond_number_b, CATEGORY_NUM])

atom_conv_b = atoms_b + tf.reduce_sum(conv_term_b, axis=2)

## convolution layer2
atoms_conv_b_reshape = tf.reshape(atom_conv_b, [-1, CATEGORY_NUM])
atoms_conv_b_reshape = tf.concat([atoms_conv_b_reshape, atoms_dummy_bulk], axis=0)

z_b_i_2 = tf.gather(atoms_conv_b_reshape, bonds_b_index1_reshape)
z_b_j_2 = tf.gather(atoms_conv_b_reshape, bonds_b_index2_reshape)
z_b_k_2 = tf.concat([z_b_i_2, z_b_j_2, bonds_b_reshape], axis=1)

sig_term_b_2 = tf.sigmoid(tf.matmul(z_b_k_2, Wf_b_2) + tf.reshape(bf_b_2, [1, CATEGORY_NUM]))
relu_term_b_2 = tf.nn.elu(tf.matmul(z_b_k_2, Ws_b_2) + tf.reshape(bs_b_2, [1, CATEGORY_NUM]))
conv_term_b_2 = tf.multiply(sig_term_b_2, relu_term_b_2)
conv_term_b_2 = tf.reshape(conv_term_b_2, [-1, atom_num_b, bond_number_b, CATEGORY_NUM])

atom_conv_b_2 = atom_conv_b + tf.reduce_sum(conv_term_b_2, axis=2)

## convolution layer3
atoms_conv_b_2_reshape = tf.reshape(atom_conv_b_2, [-1, CATEGORY_NUM])
atoms_conv_b_2_reshape = tf.concat([atoms_conv_b_2_reshape, atoms_dummy_bulk], axis=0)

z_b_i_3 = tf.gather(atoms_conv_b_2_reshape, bonds_b_index1_reshape)
z_b_j_3 = tf.gather(atoms_conv_b_2_reshape, bonds_b_index2_reshape)
z_b_k_3 = tf.concat([z_b_i_3, z_b_j_3, bonds_b_reshape], axis=1)

sig_term_b_3 = tf.sigmoid(tf.matmul(z_b_k_3, Wf_b_3) + tf.reshape(bf_b_3, [1, CATEGORY_NUM]))
relu_term_b_3 = tf.nn.elu(tf.matmul(z_b_k_3, Ws_b_3) + tf.reshape(bs_b_3, [1, CATEGORY_NUM]))
conv_term_b_3 = tf.multiply(sig_term_b_3, relu_term_b_3)
conv_term_b_3 = tf.reshape(conv_term_b_3, [-1, atom_num_b, bond_number_b, CATEGORY_NUM])

atom_conv_b_3 = atom_conv_b_2 + tf.reduce_sum(conv_term_b_3, axis=2)

## surface part
atoms_dummy_surface = tf.zeros([1, CATEGORY_NUM], dtype=tf.float32)

atoms_s_reshape = tf.reshape(atoms_s, [-1, CATEGORY_NUM])
atoms_s_reshape = tf.concat([atoms_s_reshape, atoms_dummy_surface], axis=0)
bonds_s_reshape = tf.reshape(bonds_s, [-1, BOND_CATEGORY_NUM])
bonds_s_index1_reshape = tf.reshape(bonds_s_index1, [-1, ])
bonds_s_index2_reshape = tf.reshape(bonds_s_index2, [-1, ])

with tf.name_scope("surface"):
    Wf_s_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution_filter_f')
    bf_s_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution_bias_f')
    Ws_s_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution_filter_s')
    bs_s_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution_bias_s')

    Wf_s_2 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution2_filter_f')
    bf_s_2 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution2_bias_f')
    Ws_s_2 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution2_filter_s')
    bs_s_2 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution2_bias_s')

    Wf_s_3 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution3_filter_f')
    bf_s_3 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution3_bias_f')
    Ws_s_3 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution3_filter_s')
    bs_s_3 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=initial_std), dtype=tf.float32,
                         name='Surface_convolution3_bias_s')

## convolution layer1
z_s_i_1 = tf.gather(atoms_s_reshape, bonds_s_index1_reshape)
z_s_j_1 = tf.gather(atoms_s_reshape, bonds_s_index2_reshape)
z_s_1 = tf.concat([z_s_i_1, z_s_j_1, bonds_s_reshape], axis=1)

sig_term_s = tf.sigmoid(tf.matmul(z_s_1, Wf_s_1) + tf.reshape(bf_s_1, [1, CATEGORY_NUM]))
relu_term_s = tf.nn.elu(tf.matmul(z_s_1, Ws_s_1) + tf.reshape(bs_s_1, [1, CATEGORY_NUM]))
conv_term_s = tf.multiply(sig_term_s, relu_term_s)
conv_term_s = tf.reshape(conv_term_s, [-1, atom_num_s, bond_number_s, CATEGORY_NUM])

atom_conv_s = atoms_s + tf.reduce_sum(conv_term_s, axis=2)

## convolution layer2
atoms_conv_s_reshape = tf.reshape(atom_conv_s, [-1, CATEGORY_NUM])
atoms_conv_s_reshape = tf.concat([atoms_conv_s_reshape, atoms_dummy_surface], axis=0)

z_s_i_2 = tf.gather(atoms_conv_s_reshape, bonds_s_index1_reshape)
z_s_j_2 = tf.gather(atoms_conv_s_reshape, bonds_s_index2_reshape)
z_s_2 = tf.concat([z_s_i_2, z_s_j_2, bonds_s_reshape], axis=1)

sig_term_s_2 = tf.sigmoid(tf.matmul(z_s_2, Wf_s_2) + tf.reshape(bf_s_2, [1, CATEGORY_NUM]))
relu_term_s_2 = tf.nn.elu(tf.matmul(z_s_2, Ws_s_2) + tf.reshape(bs_s_2, [1, CATEGORY_NUM]))
conv_term_s_2 = tf.multiply(sig_term_s_2, relu_term_s_2)
conv_term_s_2 = tf.reshape(conv_term_s_2, [-1, atom_num_s, bond_number_s, CATEGORY_NUM])
atom_conv_s_2 = atom_conv_s + tf.reduce_sum(conv_term_s_2, axis=2)

## convolution layer3
atom_conv_s_2_reshape = tf.reshape(atom_conv_s_2, [-1, CATEGORY_NUM])
atom_conv_s_2_reshape = tf.concat([atom_conv_s_2_reshape, atoms_dummy_surface], axis=0)

z_s_i_3 = tf.gather(atom_conv_s_2_reshape, bonds_s_index1_reshape)
z_s_j_3 = tf.gather(atom_conv_s_2_reshape, bonds_s_index2_reshape)
z_s_3 = tf.concat([z_s_i_3, z_s_j_3, bonds_s_reshape], axis=1)

sig_term_s_3 = tf.sigmoid(tf.matmul(z_s_3, Wf_s_3) + tf.reshape(bf_s_3, [1, CATEGORY_NUM]))
relu_term_s_3 = tf.nn.elu(tf.matmul(z_s_3, Ws_s_3) + tf.reshape(bs_s_3, [1, CATEGORY_NUM]))
conv_term_s_3 = tf.multiply(sig_term_s_3, relu_term_s_3)
conv_term_s_3 = tf.reshape(conv_term_s_3, [-1, atom_num_s, bond_number_s, CATEGORY_NUM])
atom_conv_s_3 = atom_conv_s_2 + tf.reduce_sum(conv_term_s_3, axis=2)

atom_pool_b = tf.div(tf.reduce_sum(atom_conv_b_3, axis=1), tf.reshape(atom_number_b, [-1, 1]))
atom_pool_s = tf.div(tf.reduce_sum(atom_conv_s_3, axis=1), tf.reshape(atom_number_s, [-1, 1]))
atom_pool = tf.concat([atom_pool_b, atom_pool_s], axis=1)

##fully connected layers
Wl_1 = tf.Variable(tf.random_normal([2 * CATEGORY_NUM, hidden_node], stddev=initial_std), name='Fully_Connected_weight1')
bl_1 = tf.Variable(tf.random_normal([hidden_node], stddev=initial_std), name='Fully_connected_bias1')

atom_fcn = tf.matmul(atom_pool, Wl_1) + bl_1
atom_fcn = tf.nn.elu(atom_fcn)
atom_fcn = tf.nn.dropout(atom_fcn, keep_prob)

Wl_2 = tf.Variable(tf.random_normal([hidden_node, 4], stddev=initial_std), name='Fully_connected_weight2')
bl_2 = tf.Variable(tf.random_normal([4], stddev=initial_std), name='Fully_connected_bias2')

v = tf.add(tf.matmul(atom_fcn, Wl_2), bl_2)

w = tf.Variable(tf.random_normal([4, 1], stddev=initial_std), name='Fully_connected_weight3')
b = tf.Variable(tf.random_normal([1], stddev=initial_std), name='Fully_connected_bias3')

y_pred = tf.matmul(v, w) + b

## cost function definition
with tf.name_scope("cost"):
    regularizer = tf.nn.l2_loss(Wl_1)
    cost = tf.losses.absolute_difference(y, y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    tf.summary.scalar("cost", cost)

acc_mse = tf.losses.mean_squared_error(y, y_pred)
acc_mae = tf.losses.absolute_difference(y, y_pred)

saver = tf.train.Saver()

ep = np.arange(epochs)

best_training_costs = []
best_test_costs = []
best_test_rmse = 100

print("data processing")

test_bulk_num, test_atom_b, test_bond_b, test_bond_index1_b, test_bond_index2_b, test_atom_num_b, test_bond_num_b, \
test_surface_num, test_atom_s, test_bond_s, test_bond_index1_s, test_bond_index2_s, test_atom_num_s, test_bond_num_s = data_preparation(
    test_materials)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(saver_path))
    
    test_y = np.reshape(test_y_data, [-1, 1])

    feed_dict_test = {atom_num_b: test_bulk_num, atom_num_s: test_surface_num, \
                      atoms_b: test_atom_b, bonds_b: test_bond_b, atom_number_b: test_atom_num_b,
                      bond_number_b: test_bond_num_b, \
                      bonds_b_index1: test_bond_index1_b, bonds_b_index2: test_bond_index2_b, \
                      atoms_s: test_atom_s, bonds_s: test_bond_s, atom_number_s: test_atom_num_s,
                      bond_number_s: test_bond_num_s, \
                      bonds_s_index1: test_bond_index1_s, bonds_s_index2: test_bond_index2_s, \
                      y: test_y, keep_prob: 1.0,}

    pred_ads_value = sess.run(y_pred, feed_dict=feed_dict_test)
    mae = sess.run(acc_mae, feed_dict=feed_dict_test)
    rmse = np.sqrt(sess.run(acc_mse, feed_dict=feed_dict_test))

    print("MAE : %.5f RMSE : %.5f" % (mae, rmse))

with open(result_file, 'w') as f:
    for i in range(len(test_materials)):
        f.write(test_materials[i] + ' ' + str(pred_ads_value[i][0]))

plt.figure(1)
x = np.arange(-8, 3, 0.1)
plt.plot(x, x, 'k--')
plt.plot(test_y_data, pred_ads_value[:, 0], 'ro', markersize=4)
plt.xlim(-8.0, 3.0)
plt.ylim(-8.0, 3.0)
plt.title("Test")
plt.xlabel('Adsorption E(DFT)')
plt.ylabel('Adsorption E(SGCNN)')

plt.show()
