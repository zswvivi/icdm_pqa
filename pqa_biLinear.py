import os
import json
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
import numpy as np
import sklearn
import pickle
#import matplotlib.pyplot as plt
# evaluation metrics
import tensorflow.keras.backend as K
from sklearn import metrics
import pandas as pd

# keras layers
# tf.keras.models.Model
# tf.keras.layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, TimeDistributed,Lambda
from tensorflow.keras.layers import GlobalAveragePooling1D,Flatten,concatenate
from tensorflow.keras.layers import Permute, multiply, RepeatVector, Activation, dot
from tensorflow.keras.layers import GlobalMaxPooling1D, subtract, Reshape, Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint

# Visualization
from tensorflow.keras.utils import plot_model
#from keras.utils.vis_utils import model_to_dot
#from IPython.display import SVG

from datetime import datetime
#now = datetime.datetime.now()
#date=str(now.year)+str(now.month)+str(now.day)

from input_fn_filter import input_fn

import sys
params = sys.argv

#os.path.join(home_path, 'dataset_params.json')
home_path = ''
data_path = ''
categories = ['Automotive','Cell_Phones_and_Accessories','Sports_and_Outdoors',
              'Tools_and_Home_Improvement','Health_and_Personal_Care',
              'Home_and_Kitchen','Patio_Lawn_and_Garden']

crq = 'sigmoid'
cate = int(params[1])

batch_size = int(params[2])
model_type='biLinearA_'+params[2]

epochs = 10
buffer_size = 1280
sentence_length  = 30
number_of_reviews= 300
number_of_nonAnswers =10
number_of_input_sentences = 2+number_of_nonAnswers+number_of_reviews
sing_nonAnswer=True

embedding_dimension = 300
numbers_of_answers = 2

# category-based parameters delcaration
category=para_json=word_vectors_file=path_words=train=dev=test=emb_matirx =  None
dataset=eval_dataset=para=train_size=dev_size=test_size=vocab_size=steps_per_epoch=validation_steps = None


def set_parameters(cate):
    global category,para_json,word_vectors_file,path_words,train,dev,test,vocab_size
    global emb_matirx,dataset,eval_dataset,para,train_size,dev_size,test_size,steps_per_epoch,validation_steps
    category =  categories[cate]
    print('Training for category: '+category)
    para_json = data_path+category+'_dataset_params.json'
    #word_vectors_file = data_path+category+fre+'_word_vec.txt'
    word_vectors_file = data_path+'word_vec.txt'

    #path_words = data_path+category+fre+'_words.txt'
    path_words = data_path + 'words.txt'

    train=data_path+category+'_train.txt'
    dev=data_path+category+'_dev.txt'
    test =data_path+category+'_test.txt'
    
    # load pre-trained wording embeddings
    with open(word_vectors_file, 'rb') as f:
        emb_matirx = pickle.load(f)

    with open(para_json) as json_data:
        para = json.load(json_data)
    
    train_size = para['train_size']
    dev_size = para['dev_size']
    dev_size = min(dev_size,11000)
    test_size = para['test_size']
    vocab_size = len(emb_matirx) #para['vocab_size']
    steps_per_epoch = round(train_size/batch_size)
    validation_steps = 1 #round(dev_size/batch_size)

    dataset = input_fn(txt=train,path_words=path_words,buffer_size=buffer_size,
                       batch_size=batch_size,number_of_reviews=number_of_reviews,
                       num_inputs=number_of_input_sentences,num_threads=16,
                       sing_nonAnswer=sing_nonAnswer,maximum_length=sentence_length)
    eval_dataset = input_fn(txt=dev,path_words=path_words,buffer_size=dev_size,
                        batch_size=dev_size,number_of_reviews=number_of_reviews,
                        num_inputs=1000+2+number_of_reviews,num_threads=16,maximum_length=sentence_length,
                        sing_nonAnswer=sing_nonAnswer,isShuffle=False,selected_nonAnswers=1)

def model_fn(embedding_dimension=embedding_dimension, vocab_size=vocab_size,
             numbers_of_answers=numbers_of_answers,number_of_reviews=number_of_reviews,
             sentence_length=sentence_length,emb_weights=emb_matirx):
    
    total_number_of_sentences=numbers_of_answers+number_of_reviews+1
    
    # INPUT + SENTENCE ENCODER
    input_layer = Input(shape=(total_number_of_sentences,sentence_length))
    emb = tf.keras.Sequential([Embedding(vocab_size,embedding_dimension,input_length=sentence_length,
                                         weights=emb_weights,
                                         trainable=True,
                                         name='emb'),
                               Dropout(0.6),
                               GlobalAveragePooling1D()])
    dis_emb = TimeDistributed(emb, input_shape=(total_number_of_sentences,sentence_length),name='dis_emb')(input_layer)
                                         
                                         
    Dense_f1 = Dense(embedding_dimension)
    #Dense_f2 = Dense(embedding_dimension)
                                         
                                         
    # split matrix into query(q) answers(a) and reviews(r)
    q = Lambda(lambda x: x[:,0:1,:], output_shape=(1,embedding_dimension),name='query')(dis_emb)
    a = Lambda(lambda x: x[:,1:(numbers_of_answers+1),:], output_shape=(numbers_of_answers,embedding_dimension),
                                                    name='answer')(dis_emb)
    r = Lambda(lambda x: x[:,(numbers_of_answers+1):,:], output_shape=(number_of_reviews,embedding_dimension),
                                                    name='review')(dis_emb)
                                         
    # C(r|q)
    q = Flatten()(q)
    q = Dense_f1(q)
    q = Reshape((embedding_dimension,1))(q)
    C_r_q = dot([r,q],axes=(2,1))
    C_r_q = Flatten()(C_r_q)
    C_r_q = Activation(crq,name='crq')(C_r_q)
                                         
                                         
    # E(a|r)
    a = TimeDistributed(Dense_f1,input_shape=(numbers_of_answers,embedding_dimension))(a)
    E_a_r = dot([r,a],axes=(2,2))
    E_a_r = Activation('sigmoid',name='ear')(E_a_r)
                                         
    # score(a|q)
    C_r_q = Reshape((number_of_reviews,1))(C_r_q)
    score = dot([E_a_r,C_r_q],axes=(1,1))  #(1,1)
    score = Flatten(name='score')(score)
                                         
    # fetch the score of true answer and the score of the highest fake answer
    score_true = Lambda(lambda x: x[:,0:1], output_shape=(1,))(score)
    score_fake = Lambda(lambda x: x[:,1:], output_shape=(numbers_of_answers-1,))(score)
    higest_fake = Lambda(lambda x: tf.nn.top_k(x)[0], output_shape=(1,))(score_fake)
    out = subtract([score_true,higest_fake])
                                         
    model = Model(inputs=input_layer, outputs=out)
    return model

# loss function
def max_loss(y_true, y_pred):
    delta=0.5
    return tf.reduce_mean(tf.maximum(0.0, delta - y_pred))


def binary_accuracy(y_true, y_pred):
    y_pred = y_pred.numpy()
    batch_size = y_pred.shape[0]
    y_true = np.array([1]*batch_size)
    y_pred = [1 if item>0 else 0 for item in y_pred]
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def one_nonAnswer(t,k=0):
    q = t[0:2] # query and true answer
    a = t[k+2:k+3] # 1 non answer
    r = t[1002:1102] # 1000 reviews
    temp=tf.concat([q,a,r],0)
    return temp


set_parameters(cate)
model = model_fn(embedding_dimension=embedding_dimension, vocab_size=vocab_size,
                    numbers_of_answers=numbers_of_answers,number_of_reviews=number_of_reviews,
                    sentence_length=sentence_length,emb_weights=emb_matirx)
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss=max_loss,
                  metrics=[binary_accuracy])

#sess = K.get_session()
#sess.run([tf.global_variables_initializer(),tf.tables_initializer()])

history = model.fit(dataset,epochs=epochs,steps_per_epoch=steps_per_epoch,
                        validation_data=eval_dataset,validation_steps=validation_steps)
history_file = os.path.join(home_path, 'experiments/results/'+category+'_'+model_type+'_history.json')
with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
model.save(filepath=os.path.join(home_path, 'experiments/results/'+category+'_'+model_type+'.h5'))

# fetching the top 5 reviews
dev_data = pd.read_csv(data_path+category+bert+'_dev.txt', sep='\t',header=None)
iterator = eval_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
layer_name='crq'
intermediate_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(next_element[0]) # only input, without label
d =pd.DataFrame(columns=['question','answer','review_0','review_1','review_2','review_3','review_4',
                             'output','largest_score_0','largest_score_1','largest_score_2','largest_score_3',
                             'largest_score_4'])
pre = model.predict(next_element[0], batch_size=dev_size)
prediction=pre.tolist()

def extract_review(i,top_k):
    x=intermediate_output[i,:]
    #index_of_largest = np.argmax(x)
    indexes = np.argsort(-x)[:top_k]
    indexes = indexes.tolist()
    
    question = dev_data.iloc[i,0]
    answer = dev_data.iloc[i,1]
    reviews = dev_data.iloc[i,1002:]  # all reviews
    
    relevant_review = [reviews.iloc[j] for j in indexes]
    largest_softmax = [x[j] for j in indexes]
    
    return [question,answer,relevant_review,largest_softmax]

for i in range(dev_size):
            t = extract_review(i,top_k=5)
            d = d.append({'question':t[0],'answer':t[1],'review_0':t[2][0],'review_1':t[2][1],'review_2':t[2][2],
                      'review_3':t[2][3],'review_4':t[2][4],'output':prediction[i][0],
                      'largest_score_0':t[3][0],'largest_score_1':t[3][1],'largest_score_2':t[3][2],
                     'largest_score_3':t[3][3],'largest_score_4':t[3][4]},ignore_index=True)
#d = d.sort_values(by=['output'], ascending=False)
d.to_csv(home_path+'/experiments/results/'+category+'_'+model_type+'_top5_reviews.csv', index=None, sep='\t', mode='w')
