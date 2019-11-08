import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle

#from tensorflow import keras
import os
#import re

#import bert
import run_classifier
import optimization
import tokenization
import modeling
#import json
import ast
from os import path
import sys
params = sys.argv
#cate = int(params[1])



category = params[1]

init_checkpoint = './trained_model/FLTR/'
OUTPUT_DIR = './trained_model/'+category+'_FLTR'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data_path = './data/'+category+'.txt'

BATCH_SIZE = 180
MAX_SEQ_LENGTH = 40
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
PAD_WORD = '[PAD]'

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

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
#tokenizer.tokenize("This here's an example of using the BERT tokenizer")


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.7)
    if is_predicting:
        output_layer = tf.nn.dropout(output_layer, keep_prob=1)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
          
      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      checkpoint_file = OUTPUT_DIR+'/checkpoint'
             
      if path.exists(checkpoint_file) is False:
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
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        return {
            "eval_accuracy": accuracy
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn



def QR_pair(t):
    temp=[]
    num_reviews = t['num_reviews']
    for i in range(num_reviews):
        temp.append([t['question'],t['reviewText'][i]])
    return temp


def bert_ranker(d,estimator):
    d=d.reset_index(drop=True)
    test_t = d.apply(QR_pair,axis=1)
    test_t = test_t.tolist()
    flat_list = [item for sublist in test_t for item in sublist]
    test_t = pd.DataFrame(flat_list,columns=['question','review'])
    test_t['question'] = test_t['question'].apply(str)
    test_t['review'] = test_t['review'].apply(str)
    DATA_COLUMN_A = 'question'
    DATA_COLUMN_B = 'review'
    max_inputs = 1000000
    probs = []
    temp_test = test_t.copy()
    
    while len(temp_test)>0:
        line = min(max_inputs,len(temp_test))
        temp = temp_test[:line]
        
        inputExamples = temp.apply(lambda x: run_classifier.InputExample(guid=None,
                                    text_a = x[DATA_COLUMN_A],
                                    text_b = x[DATA_COLUMN_B],
                                    label = 0), axis = 1)
                                    
        input_features = run_classifier.convert_examples_to_features(inputExamples, label_list,
                                            MAX_SEQ_LENGTH, tokenizer)
                                            
        
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                         is_training=False, drop_remainder=False)
        predictions = estimator.predict(predict_input_fn)
        probabilities = [prediction['probabilities'] for prediction in  predictions]
        probs = probs+[item.tolist()[1] for item in probabilities]
        
        if len(temp_test)>max_inputs:
            temp_test = temp_test[line:]
            temp_test = temp_test.reset_index(drop=True)
        else:
            temp_test = []

    test_t['probilities']=probs
    num_reviews = d['num_reviews'].tolist()
    d['FLTR_scores'] = ''
    for i in range(0,len(d)):
        n = num_reviews[i]
        #print(probs[:n])
        d.at[i,'FLTR_scores'] = probs[:n]
        #print(d.at[i,'FLTR_scores'])
        if i!=len(d)-1:
            probs=probs[n:]
            
    return d
    

print('Working on '+category)
print('Reading datasets..')
current_time = datetime.now()
data = pd.read_csv(data_path,sep='\t',encoding='utf-8',#nrows=100,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval})

data['question'] = data['QA'].apply(lambda x: x['questionText'])
data['answer'] = data['QA'].apply(lambda x: x['answers'][0]['answerText'] if len(x['answers'])>0 else PAD_WORD)
data['num_reviews']= data['reviewText'].apply(lambda x: len(x))

train = data[:int(len(data)*0.8)]
dev = data[int(len(data)*0.8):int(len(data)*0.9)]
test = data[int(len(data)*0.9):]

# fine-tuning FLTR for each category
list_of_answers = list(train['answer'])
list_of_answers=shuffle(list_of_answers)
qa = train[['question','answer']]
nqa =  pd.DataFrame({'question': train['question'].tolist(),'answer':list_of_answers})
qa['label']=1
nqa['label']=0

d = pd.concat([qa,nqa],axis=0)
d=shuffle(d)
d['question']=d['question'].apply(str)
d['answer']=d['answer'].apply(str)
split = int(len(d)*0.9)
dtrain = d[0:split]
dtest = d[split:]

DATA_COLUMN_A = 'question'
DATA_COLUMN_B = 'answer'
LABEL_COLUMN = 'label'
label_list = [0, 1]

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = dtrain.apply(lambda x: run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                        text_a = x[DATA_COLUMN_A],
                                                                        text_b = x[DATA_COLUMN_B],
                                                                        label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = dtest.apply(lambda x: run_classifier.InputExample(guid=None,text_a = x[DATA_COLUMN_A],
                                            text_b = x[DATA_COLUMN_B],label = x[LABEL_COLUMN]), axis = 1)
                                            
# Convert our train and test features to InputFeatures that BERT understands.
train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * 2)
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

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = run_classifier.input_fn_builder(
                                                 features=train_features,
                                                 seq_length=MAX_SEQ_LENGTH,
                                                 is_training=True,
                                                 drop_remainder=True)

print(f'Beginning Training!')
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)
test_input_fn = run_classifier.input_fn_builder(
                                                features=test_features,
                                                seq_length=MAX_SEQ_LENGTH,
                                                is_training=False,
                                                drop_remainder=False)
                                                
#estimator.evaluate(input_fn=test_input_fn, steps=None)
predictions = estimator.predict(test_input_fn)
x=[prediction['labels'] for prediction in predictions]
dtest['prediction']=x
from sklearn.metrics import accuracy_score
print('The accuracy of fine-tuning FLTR is: '+str(accuracy_score(dtest.label,dtest.prediction)))


# FLTR ranking reivews
train=bert_ranker(train,estimator)
dev=bert_ranker(dev,estimator)
test=bert_ranker(test,estimator)
data=pd.concat([train,dev,test],axis=0,ignore_index=True)
#data['question'] = d['question'].apply(lambda x: str(x).replace('\t',''))
#data['answer'] = d['answer'].apply(lambda x: str(x).replace('\t',''))
      
if(data.isnull().values.any()):
    data = data.replace(np.nan, PAD_WORD, regex=True)
    
data.to_csv(data_path, index=None, sep='\t', mode='w')
print("FLTR took time ", datetime.now() - current_time)

