import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle

from tensorflow import keras
import os
import re

#import bert
import run_classifier
import optimization
import tokenization
import modeling
import json
import ast
import copy
from os import path
import sys
params = sys.argv

category = params[1]

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
init_checkpoint = './data/ALL/'

OUTPUT_DIR = '/scratch1/zha274/QAExperiment/'+category
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  
data_path = '/scratch1/zha274/QAdata/'+category+'.txt'
    
MAX_SEQ_LENGTH = 40
BATCH_SIZE = 180
num_reviews= 10
Big_Batch = int(BATCH_SIZE/(3*num_reviews))

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })
    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    d = d.prefetch(batch_size)
    if is_training:
        d = d.repeat()
    return d

  return input_fn

def create_model(is_predicting, features,num_labels,k=num_reviews*3,batch_size=BATCH_SIZE):
    hidden_size =768    
    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [1, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [1], initializer=tf.zeros_initializer())
    
    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    
    label_ids = features["label_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
 
    bert_outputs = bert_module(
              inputs=bert_inputs,
              signature="tokens",
              as_dict=True)     
    
    output_layer = bert_outputs["pooled_output"]
    #output_layer = tf.Print(output_layer,[output_layer,tf.shape(output_layer)],"outputlayer:")
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.7)
    if is_predicting:
        output_layer = tf.nn.dropout(output_layer, keep_prob=1)
        
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits,[int(batch_size/k),k])
    #logits = tf.Print(logits,[logits,tf.shape(logits)],'logits:')

    def scoring_fn(temp):
        rq = temp[0:num_reviews]
        rq = tf.reshape(rq,[num_reviews])
        rq = tf.nn.softmax(rq, axis=-1)
        #rq = tf.Print(rq,[rq],"Score of each review to question:")
    
        ar = temp[num_reviews:num_reviews*2]
        ar = tf.reshape(ar,[num_reviews])
        ar = tf.sigmoid(ar)
        #ar = tf.Print(ar,[ar],"Score of each review to true answer:")
    
        nar = temp[num_reviews*2:num_reviews*3]
        nar = tf.reshape(nar,[num_reviews]) 
        nar = tf.sigmoid(nar)
        #nar = tf.Print(nar,[nar],"Score of each review to non-answer:")
    
        ars = tf.tensordot(rq, ar, 1)
        nars = tf.tensordot(rq, nar, 1)
    
        #ars = tf.Print(ars,[ars,tf.shape(ars)],'Score of ture answer: ')
        #nars = tf.Print(nars,[nars,tf.shape(nars)],'Score of non-answer: ')
        
        score = tf.subtract(ars,nars)
        #score = tf.Print(score,[score],'Score: ')
        return score,rq
    
    scores,rqs = tf.map_fn(scoring_fn,logits,dtype=(tf.float32, tf.float32))
    #scores = tf.Print(scores,[scores,tf.shape(scores)],'scores: ')
    
    with tf.variable_scope("loss"):    
        #rqs = tf.Print(rqs,[rqs],"p(review | question):")
        predicted_labels = tf.squeeze(tf.round(tf.to_float(scores)+0.5))         
        delta = 0.5       
        per_example_loss = tf.maximum(0.0, delta - scores)
        per_example_loss = tf.Print(per_example_loss,[per_example_loss],'per_example_loss: ')
        loss = tf.reduce_mean(per_example_loss)
        loss = tf.Print(loss,[loss],'loss: ')
        if is_predicting:
              return (predicted_labels, scores,rqs)
        return (loss, predicted_labels, scores)

    
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    #print(label_ids)
    #print(features)
    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, features, num_labels)
    
      predicted_labels = tf.Print(predicted_labels,[predicted_labels],'predicted_labels: ')
#           (loss, predicted_labels, log_probs) = create_model(
#         is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
      #print(predicted_labels)
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
      
      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      checkpoint_file = OUTPUT_DIR+'/checkpoint'
             
      if path.exists(checkpoint_file) is False:
         if path.exists(init_checkpoint+'checkpoint') is True:
             (assignment_map, initialized_variable_names,init_vars
                     ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

             tf.logging.info("**** Trainable Variables ****")
             for var in tvars:
                     init_string = ""
                     if var.name in initialized_variable_names:
                       init_string = ", *INIT_FROM_CKPT*"
                     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                 init_string)
                   
             tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
       
      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        predicted_labels = tf.cast(predicted_labels,tf.int32)
        #accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        accuracy =tf.metrics.mean(tf.equal(label_ids[0:Big_Batch], predicted_labels))
        return {
            "eval_accuracy": accuracy
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          eval_metric_ops=eval_metrics,
          train_op=train_op)
      else:
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs,rqs) = create_model(
        is_predicting,features, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels,
          'crq':rqs
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

def FLTR_Top10(t):
    reviews = copy.deepcopy(t['reviewText'])
    FLTR_scores = copy.deepcopy(t['FLTR_scores'])
    temp=[]
    for i in range(0,num_reviews):
        if i<len(reviews):
            index = FLTR_scores.index(max(FLTR_scores))
            temp.append(reviews[index])
            reviews.pop(index)
            FLTR_scores.pop(index)
        else:
            temp.append('[PAD]')
    return temp

print('Working on '+category)
print('Reading datasets..')
current_time = datetime.now()
categories = ['Tools_and_Home_Improvement','Patio_Lawn_and_Garden','Automotive','Cell_Phones_and_Accessories','Health_and_Personal_Care','Sports_and_Outdoors','Home_and_Kitchen']

data = []
if category == 'ALL':)
    for cate in categories:
        each_path = '/scratch1/zha274/QAdata/'+cate+'.txt'
        temp = pd.read_csv(each_path,sep='\t',encoding='utf-8',#nrows=100,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval,'FLTR_scores':ast.literal_eval})
        train = data[:int(len(temp)*0.8)]
        data.append(temp)
    data =pd.concat(data,axis=0)

else:
    data = pd.read_csv(data_path,sep='\t',encoding='utf-8',#nrows=100,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval,'FLTR_scores':ast.literal_eval})

    
data['FLTR_Top10'] = data.apply(FLTR_Top10,axis=1)

list_of_answers = list(data['answer'])
list_of_answers=shuffle(list_of_answers)
data['non_answer']= list_of_answers

train = data[:int(len(data)*0.8)]
dev = data[int(len(data)*0.8):int(len(data)*0.9)]
test = data[int(len(data)*0.9):]
 
def qar_pair(t):
    x1=[]
    x2=[]
    x3=[]
    for i in range(num_reviews):
        x1.append([t['question'],t['FLTR_Top10'][i]])
        x2.append([t['answer'],t['FLTR_Top10'][i]])    
        x3.append([t['non_answer'],t['FLTR_Top10'][i]])
    return x1+x2+x3

def BerQA_train_predict(data,is_training=True):
    d=data.copy()
    scores = []
    max_inputs = 30000
    
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 2
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    SAVE_SUMMARY_STEPS = 500
    num_train_steps = 100
    max_steps = 100000
    
    DATA_COLUMN_A = 'senA'
    DATA_COLUMN_B = 'senB'
    LABEL_COLUMN = 'Label'
    label_list = [0, 1]
    
    while(len(d)>0 and num_train_steps<=max_steps):
        line = min(max_inputs,len(d))
        temp = d[:line]
        temp_t = temp.apply(qar_pair,axis=1)
        temp_t=temp_t.tolist()
        flat_list = [item for sublist in temp_t for item in sublist]
        temp_t =pd.DataFrame(flat_list,columns=['senA','senB'])	
        temp_t['Label'] =1
        temp_t['senA']=temp_t['senA'].apply(str)
        temp_t['senB']=temp_t['senB'].apply(str)

        temp_InputExamples = temp_t.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                           text_a = x[DATA_COLUMN_A],
                                                                           text_b = x[DATA_COLUMN_B],
                                                                           label = x[LABEL_COLUMN]), axis = 1)
        
        temp_features = run_classifier.convert_examples_to_features(temp_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
        num_train_steps = int(len(temp_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)+num_train_steps
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        # Specify outpit directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
            model_dir=OUTPUT_DIR,
            save_summary_steps=SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

        model_fn = model_fn_builder(
            num_labels=len(label_list),
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": BATCH_SIZE})

        input_fn = input_fn_builder(
            features=temp_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=is_training,
            drop_remainder=True)
            
        if is_training:
            print('Beginning Training!')
            early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(estimator,metric_name='loss',max_steps_without_decrease=1000,min_steps=100)
            current_time = datetime.now()
            #tf.estimator.train_and_evaluate(estimator,train_spec=tf.estimator.TrainSpec(input_fn, hooks=[early_stopping]))
            estimator.train(input_fn=input_fn, max_steps=num_train_steps,hooks=[early_stopping])
            print("Training took time ", datetime.now() - current_time)
        else:
            predictions = estimator.predict(input_fn)
            outputs=[(prediction['probabilities'],prediction['crq']) for prediction in predictions]
            x=[i[0] for i in outputs]
            y=[i[1] for i in outputs]
            print('\n')
            print('Accuracy of '+category+' is: '+str(sum(i > 0 for i in x)/len(x)))
            print('\n')
            scores = scores+y

        if len(d)>max_inputs:
            d = d[line:]
            d = d.reset_index(drop=True)
        else:
            d = []
            
    if is_training is False:
        data = data[:len(scores)]
        scores = [item.tolist() for item in scores]
        #BERTQA_scores = pd.DataFrame(data=scores)
        data['BERTQA_scores'] = scores
        #data = pd.concat([data,BERTQA_scores],axis=1,ignore_index=True)
        return data


BerQA_train_predict(train)
train_outputs = BerQA_train_predict(train,is_training=False)
dev_outputs = BerQA_train_predict(dev,is_training=False)
test_outputs = BerQA_train_predict(test,is_training=False)

data=pd.concat([train_outputs,dev_outputs,test_outputs],axis=0,ignore_index=True)
data=data.drop(columns=['non_answer'])
if(data.isnull().values.any()):
    data = data.replace(np.nan, '[PAD]', regex=True)
    
data.to_csv(data_path, index=None, sep='\t', mode='w')
print("BERTQA took time ", datetime.now() - current_time)
