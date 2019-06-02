# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:13:59 2019

@author: SRIKANT
"""

from create_featuresets import create_feature_sets_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_nodes_hl4 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

keep_prob = tf.placeholder('float')
x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}
                                                      
hidden_4_layer = {'f_fum':n_nodes_hl4,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl4]))}                                                

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}



def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.dropout(l1,keep_prob)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.dropout(l2,keep_prob)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3,keep_prob)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weight']), hidden_4_layer['bias'])
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.dropout(l4,keep_prob)

    output = tf.matmul(l4,output_layer['weight']) + output_layer['bias']
    
    output= tf.nn.dropout(output,keep_prob)
    return output

def train_neural_network(x):
    
    prediction = neural_network_model(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.3).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y,keep_prob:0.4})
                epoch_loss += c
                i+=batch_size
        
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        
        correct = tf.equal(tf.argmax(prediction, 1),tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        


        print('Accuracy:',accuracy.eval({x:test_x, y:test_y,keep_prob:0.5}))
        
        
           
            
	    
train_neural_network(x)



