from sklearn.model_selection import train_test_split
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
import json
import ast
import sys
#params = sys.argv
#FLTR = int(params[1])

OUTPUT_DIR = '/scratch1/zha274/QAExperiment/FLTR'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

data_path = '/scratch1/zha274/QAdata/ALL.txt'
BATCH_SIZE = 180
MAX_SEQ_LENGTH = 40
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 20.0

# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 1000
PAD_WORD = '[PAD]'


# This is a path to an uncased (all lowercase) version of BERT
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

# Crossing-training FLTR
ALL = pd.read_csv(data_path,sep='\t',encoding='utf-8',#nrows=1000,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval})

ALL = shuffle(ALL)
ALL['question'] = ALL['QA'].apply(lambda x: x['questionText'])
ALL['answer'] = ALL['QA'].apply(lambda x: x['answers'][0]['answerText'] if len(x['answers'])>0 else PAD_WORD)
list_of_answers = list(ALL['answer'])
list_of_answers=shuffle(list_of_answers)

qa = ALL[['question','answer']]
nqa =  pd.DataFrame({'question': ALL['question'].tolist(),'answer':list_of_answers})
qa['label']=1
nqa['label']=0

data = pd.concat([qa,nqa],axis=0)
data=shuffle(data)
data['question']=data['question'].apply(str)
data['answer']=data['answer'].apply(str)
split = int(len(data)*0.9)
train = data[0:split]
test = data[split:]
DATA_COLUMN_A = 'question'
DATA_COLUMN_B = 'answer'
LABEL_COLUMN = 'label'
label_list = [0, 1]

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                        text_a = x[DATA_COLUMN_A],
                                                                        text_b = x[DATA_COLUMN_B],
                                                                        label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                      text_a = x[DATA_COLUMN_A],
                                                                      text_b = x[DATA_COLUMN_B],
                                                                      label = x[LABEL_COLUMN]), axis = 1)

# Convert our train and test features to InputFeatures that BERT understands.
train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
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
current_time = datetime.now()
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
test['prediction']=x
from sklearn.metrics import accuracy_score
print('The accuracy of  cross-domian training FLTR is: '+str(accuracy_score(test.label,test.prediction)))
