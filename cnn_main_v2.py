# version 2

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from readfunc_v2 import single_dataset_gen
from readfunc_v2 import examples
from readfunc_v2 import kfold_dataset_gen


tf.logging.set_verbosity(tf.logging.INFO)
def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=5)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(5,5), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op

def cal_p_r_fscore(conf):
    FP = np.sum(conf, axis=0) - np.diag(conf)
    FN = np.sum(conf, axis=1) - np.diag(conf)
    TP = np.diag(conf)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Precision:", precision)
    print("Recall:", recall)
    fscore = 2 * precision * recall / (precision + recall)
    print("F-Score:", fscore)

def cnn_model_fn(features, labels, mode):
  # in,c1,p1,c2,p2,flat,dense,drop,logits

  input_layer = tf.reshape(features["x"], [-1, 120, 120, 3]) # -1,28,28,1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 30 * 30 * 64 * 1])
  dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=5)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1,name="predication_classes"),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #loss
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  # Add evaluation metrics (for EVAL mode)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "conf_matrix": eval_confusion_matrix(
          labels, predictions["classes"])
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def gen_input_fn(features, labels, bs=32, ep=2, sh=True):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=labels,
        batch_size=bs,
        num_epochs=ep,
        shuffle=sh)
    return input_fn

def kfold_cross_validation(data):
    kf = KFold(n_splits=10)
    for train_ind, eval_ind in kf.split(data):
        train_d = examples(len(train_ind))
        evaluate_d = examples(len(eval_ind))
        train_d.images = data.images[train_ind]
        train_d.labels = data.labels[train_ind]
        evaluate_d.images = data.images[eval_ind]
        evaluate_d.labels = data.labels[eval_ind]
        return train_d, evaluate_d


# --------------- Main Processing ----------------
# Create the Estimator
print("Estimator Creating...")
action_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir="/tmp/cnn_model_no_data_aug")
  #model_fn=cnn_model_fn, model_dir="/tmp/cnn_model_data_aug")

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
print("Training Logging Setting...")
tensors_to_log = {"probabilities": "softmax_tensor",
                "classes": "predication_classes"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

# Load training, evaluation, and testing data
Flag_Once_Generate_Dateset = False

if Flag_Once_Generate_Dateset == True:
    print("Generate single dataset..")
    train1,eval1, test1 = single_dataset_gen()
    train_data = train1.images
    train_labels = train1.labels
    eval_data = eval1.images
    eval_labels = eval1.labels
    test_data = test1.images
    test_labels =  test1.labels
    print("Get Training data:", np.shape(train_data))
    print("Get Evaluating data:", np.shape(eval_data))
    print("Get Testing data:", np.shape(test_data))
    print(train_labels)



# Evaluation both training and evaluation data once
Flag_Once_Evaluation = False

if Flag_Once_Evaluation == True:
    print("--Evaluation once start--")
    eval_Tr_input_fn = gen_input_fn(train_data, train_labels, bs=32, ep=2, sh=False)
    eval_Tr_results = action_classifier.evaluate(input_fn=eval_Tr_input_fn)
    print("-----------------------")
    print("Evaluate Training data result:", eval_Tr_results)

    eval_input_fn = gen_input_fn(eval_data, eval_labels, bs=32, ep=2, sh=False)
    eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
    # print precision, recall and fscore
    cal_p_r_fscore(eval_Tr_results['conf_matrix'])
    print("-----------------------")
    print("Evaluate Evaluating data result:", eval_results)
    cal_p_r_fscore(eval_results['conf_matrix'])
    print("--Evaluation once end--")
# Cross Validation
Flag_Cross_Validation = True

if Flag_Cross_Validation == True:
    data = kfold_dataset_gen()
    kf = KFold(n_splits=10,shuffle=True)
    count = 0
    total_acc = []
    model_dir = "/tmp/cnn_model_cv"
    print("10-fold cross validation starts...")
    print("Before spliting:",len(data.labels))
    for tr_ind, ev_ind in kf.split(data.labels):
        print(len(tr_ind),len(ev_ind))
        tr = examples(len(tr_ind))
        ev = examples(len(ev_ind))
        tr.images = data.images[tr_ind]
        tr.labels = data.labels[tr_ind]
        ev.images = data.images[ev_ind]
        ev.labels = data.labels[ev_ind]
        # Create a new model
        model_dir = model_dir + '_' + str(count)
        action_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_dir)
        train_input_fn = gen_input_fn(tr.images, tr.labels, bs=32, ep=2, sh=False)
        # Train & Evaluate
        print("Model:", count, "Training start")
        action_classifier.train(
            input_fn=train_input_fn, steps=800, hooks=[logging_hook])
        eval_input_fn = gen_input_fn(ev.images, ev.labels, bs=32, ep=2, sh=False)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        print("Evaluate result:", eval_results)
        cal_p_r_fscore(eval_results['conf_matrix'])
        total_acc.append(eval_results['accuracy'])
        print(ev_ind)
        count += 1
    print("Average Acc:", sum(total_acc) / len(total_acc))


# Long 2500 steps Training
Flag_Long_Training = False

if Flag_Long_Training == True:
    acc_train = []
    acc_eval = []
    acc_test = []
    loss_train = []
    print("long long long training starts...")
    for i in range(50):
        # Train the model
        train_input_fn = gen_input_fn(train_data, train_labels, bs=32, ep=2, sh=False)
        action_classifier.train(
          input_fn=train_input_fn,
          steps=50, #20000
          hooks=[logging_hook])

        # Evaluate training data
        eval_Tr_input_fn =gen_input_fn(train_data, train_labels, bs=32, ep=2, sh=False)
        eval_Tr_results = action_classifier.evaluate(input_fn=eval_Tr_input_fn)
        print("Evaluate Training data:", eval_Tr_results)
        # Evaluate evaluating data
        eval_input_fn =  gen_input_fn(eval_data, eval_labels, bs=32, ep=2, sh=False)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        print("Evaluate Evaluating data:",eval_results)
        # log
        acc_eval.append(eval_results['accuracy'])
        acc_train.append(eval_Tr_results['accuracy'])
        loss_train.append((eval_Tr_results['loss']))
        # Testing the model by filmed video.
        test_input_fn = gen_input_fn(test_data, test_labels, bs=1, ep=2, sh=True)
        test_results = action_classifier.evaluate(input_fn=test_input_fn)
        print("Evaluate Testing data:",test_results)
        acc_test.append(test_results['accuracy'])

# The shape of each layer in CNN model
#for tmp in  action_classifier.get_variable_names():
#    print(np.shape(action_classifier.get_variable_value(name=tmp)))
