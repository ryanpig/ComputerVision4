# version 2

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from readfunc_v2_optical import single_dataset_gen
from readfunc_v2_optical import examples
from readfunc_v2_optical import kfold_dataset_gen
from Visualize import plot_acc_bar
from Visualize import plot_acc_loss

IMAGE_SIZE_RGB = 120
IMAGE_SIZE_HOG1 = 100
IMAGE_SIZE_HOG2 = 90

tf.logging.set_verbosity(tf.logging.INFO)
# Calculate CM and Convert CM to the format(value, update_op)
def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=5)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(5,5), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op
# Calculate Precision, Recall, and F-score
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
def conv2d(inputs, filters, ks, pad):

    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=ks,
        padding=pad,
        activation=tf.nn.relu)
    return conv

# CNN model function
def cnn_model_fn(features, labels, mode):

  #  Motion input (Optical Flow)
  input_in1 = tf.reshape(features["x1"], [-1, 120, 120, 20])
  #conv1_in1 = conv2d(input_in1, 32, [5, 5], "Same") #[5,5,20]
  conv1_in1 = tf.layers.conv2d(input_in1, 32,[5, 5], padding="Same", activation=tf.nn.relu)
  pool1_in1 = tf.layers.max_pooling2d(inputs=conv1_in1, pool_size=[2, 2], strides=2)
  conv2_in1 = tf.layers.conv2d(pool1_in1, 64, [5, 5], padding="Same", activation=tf.nn.relu)
  #conv2_in1 = conv2d(pool1_in1, 64, [5, 5], "Same")
  pool2_in1 = tf.layers.max_pooling2d(inputs=conv2_in1, pool_size=[2, 2], strides=2)
  pool2_flat_in1 = tf.reshape(pool2_in1, [-1, 30 * 30 * 64 * 1])
  dense_in1 = tf.layers.dense(inputs=pool2_flat_in1, units=128, activation=tf.nn.relu)
  dropout_in1 = tf.layers.dropout(inputs=dense_in1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits_in1 = tf.layers.dense(inputs=dropout_in1, units=5)
  prob_in1 = tf.nn.softmax(logits_in1, name="softmax_tensor_in1")


  #  Spatial input (RGB image)
  input_in2 = tf.reshape(features["x2"], [-1, 120, 120, 20])
    #conv1_in2 = conv2d(input_in2, 32, [5, 5], "Same")
  conv1_in2 = tf.layers.conv2d(input_in2, 32,[5, 5], padding="Same", activation=tf.nn.relu)
  pool1_in2 = tf.layers.max_pooling2d(inputs=conv1_in2, pool_size=[2, 2], strides=2)
  conv2_in2 = tf.layers.conv2d(pool1_in2, 64, [5, 5], padding="Same", activation=tf.nn.relu)
    #conv2_in2 = conv2d(pool1_in2, 64, [5, 5], "Same")
  pool2_in2 = tf.layers.max_pooling2d(inputs=conv2_in2, pool_size=[2, 2], strides=2)
  pool2_flat_in2 = tf.reshape(pool2_in2, [-1, 30 * 30 * 64 * 1])
  dense_in2 = tf.layers.dense(inputs=pool2_flat_in2, units=128, activation=tf.nn.relu)
  dropout_in2 = tf.layers.dropout(inputs=dense_in2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits_in2 = tf.layers.dense(inputs=dropout_in2, units=5)
  prob_in2 = tf.nn.softmax(logits_in2, name="softmax_tensor_in2")

  print([x.name for x in tf.global_variables()])


  # Last Fusion
  #logits = tf.divide(tf.add(logits_in1,logits_in2), 2.0)
  score_avg = tf.divide(tf.add(prob_in1,prob_in2),2.0, "softmax_tensor")
  print("Score_avg:")
  print(tf.shape(score_avg))
  print(np.shape(score_avg))
  # classified by the averaging of probabilities (softmax) or the averaging of logits
  # axis 1 -> 0, input: logits -> score_avg
  predictions = {
      "classes": tf.argmax(input=score_avg, axis=1,name="predication_classes"),
      "probabilities": score_avg
  }
  '''
  predictions = {
      "classes": tf.argmax(input=logits_in1, axis=1, name="predication_classes"),
      "probabilities": tf.nn.softmax(logits_in1, name="softmax_tensor")
  }
'''

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #loss (not sure logits or score_avg
  loss1 = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_in1)
  loss2 = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_in2)
  loss = tf.add(loss1,loss2)

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
# Generate input_fn for training, evaluating
def gen_input_fn(features, labels, bs=32, ep=2, sh=True):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": features},
        y=labels,
        batch_size=bs,
        num_epochs=ep,
        shuffle=sh)
    return input_fn
def gen_input_fn_dual(features, labels, bs=32, ep=2, sh=True):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x1": features, "x2":features},
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

def evaluate_print_result(classifier, input_fn_Tr, input_fn_Eval):
    print("--Evaluation once start--")
    eval_Tr_results = classifier.evaluate(input_fn=input_fn_Tr)
    print("-----------------------")
    print("Evaluate Training data result:", eval_Tr_results)
    eval_Eval_results = action_classifier.evaluate(input_fn=input_fn_Eval)
    # print precision, recall and fscore
    cal_p_r_fscore(eval_Tr_results['conf_matrix'])
    print("-----------------------")
    print("Evaluate Evaluating data result:", eval_Eval_results)
    cal_p_r_fscore(eval_Eval_results['conf_matrix'])
    print("--Evaluation once end--")
    acc_tr = eval_Tr_results['accuracy']
    acc_eval = eval_Eval_results['accuracy']
    return acc_tr, acc_eval
def gen_dataset(train_ratio, tr_eval_ratio):
    print("Generate single dataset..")
    train1,eval1, test1 = single_dataset_gen(train_usage_ratio=train_ratio, train_eval_ratio=tr_eval_ratio)
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
    return train_data, train_labels, eval_data, eval_labels, test_data, test_labels

# --------------- Main Processing ----------------
# Create the Estimator
print("Estimator Creating...")
action_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir="/tmp/cnn_model_two4")
  #model_fn = cnn_model_fn, model_dir = "/tmp/cnn_model_HOG4")
  #model_fn=cnn_model_fn, model_dir="/tmp/cnn_model_data_aug")

# The shape of each layer in CNN model
#for tmp in  action_classifier.get_variable_names():
#    print(np.shape(action_classifier.get_variable_value(name=tmp)))

# Logging
print("Training Logging Setting...")
tensors_to_log = {"probabilities": "softmax_tensor",
                "classes": "predication_classes"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
#--------------- Configuration ----------------
Flag_Once_Generate_Dateset = True
Flag_Once_Training = True
Flag_Once_Evaluation = True

# Configuration for testing
Flag_Cross_Validation_Test = False
Flag_Reduce_Training_Test = False
Flag_Long_Training = False

# Load training, evaluation, and testing data
if Flag_Once_Generate_Dateset == True:
    train_data, train_labels, eval_data, \
    eval_labels, test_data, test_labels = gen_dataset(train_ratio=0.8,tr_eval_ratio=0.2)

# Train the model
if Flag_Once_Training == True:
    train_input_fn = gen_input_fn_dual(train_data, train_labels, bs=32, ep=5, sh=False)
    action_classifier.train(
      input_fn=train_input_fn,
      steps=400, #20000
      hooks=[logging_hook])

# Evaluation both training and evaluation data once
if Flag_Once_Evaluation == True:
    eval_Tr_input_fn = gen_input_fn_dual(train_data, train_labels, bs=32, ep=1, sh=False)
    eval_input_fn = gen_input_fn_dual(eval_data, eval_labels, bs=32, ep=1, sh=False)
    acc_tr, acc_eval = evaluate_print_result(action_classifier,eval_Tr_input_fn, eval_input_fn )

# Cross Validation
if Flag_Cross_Validation_Test == True:
    data = kfold_dataset_gen(feature_type='RGB',train_eval_ratio=0.2)
    kf = KFold(n_splits=10,shuffle=True)
    count = 0
    total_acc = []
    model_dir1 = "/tmp/cnn_model_cv"
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
        model_dir = model_dir1 + '_' + str(count)
        action_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_dir)
        train_input_fn = gen_input_fn(tr.images, tr.labels, bs=32, ep=1, sh=False)
        # Train & Evaluate
        print("Model:", model_dir, "Training start")
        action_classifier.train(
            input_fn=train_input_fn, steps=800, hooks=[logging_hook])
        eval_input_fn = gen_input_fn(ev.images, ev.labels, bs=32, ep=1, sh=False)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        # Print Result
        print("Evaluate result:", eval_results)
        cal_p_r_fscore(eval_results['conf_matrix'])
        total_acc.append(eval_results['accuracy'])
        print(ev_ind)
        count += 1
    print("Average Acc:", sum(total_acc) / len(total_acc))
    plot_acc_bar(total_acc, 'Cross-Validation')

# Reduce training dataset test
# Running four kinds of quantitity of training examples
if Flag_Reduce_Training_Test == True:
    # four kinds of training usage ratio (e.g. 30% of whole dataset)
    train_ratios = [0.3,0.5,0.7,0.9]
    total_acc = []
    model_dir1 = "/tmp/cnn_model_reduce"
    count = 0
    print("Reduced training test starts...")
    for tr_ratio in train_ratios:
        # Generate new dataset by train_ratios
        train_data, train_labels, eval_data, \
        eval_labels, test_data, test_labels = gen_dataset(train_ratio=tr_ratio, tr_eval_ratio=0.3)
        # Create model
        model_dir = model_dir1 + '_' + str(count)
        action_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_dir)
        # Training
        print("Model:", model_dir, "Training start")
        train_input_fn = gen_input_fn(train_data, train_labels, bs=16, ep=1, sh=True)
        action_classifier.train(
          input_fn=train_input_fn,
          steps=500, #20000
          hooks=[logging_hook])
        # Evaluate
        eval_Tr_input_fn = gen_input_fn(train_data, train_labels, bs=32, ep=1, sh=False)
        eval_input_fn = gen_input_fn(eval_data, eval_labels, bs=32, ep=1, sh=False)
        acc_tr, acc_eval = evaluate_print_result(action_classifier, eval_Tr_input_fn, eval_input_fn)
        total_acc.append(acc_eval)
        count += 1
    print("Average Acc:", sum(total_acc) / len(total_acc))
    plot_acc_bar(total_acc, 'Four kinds of training ratio')

# Long 2500 steps Training
if Flag_Long_Training == True:
    acc_train = []
    acc_eval = []
    acc_test = []
    loss_train = []
    print("long long long training starts...")
    # Generate new dataset by train_ratios
    train_data, train_labels, eval_data, \
    eval_labels, test_data, test_labels = gen_dataset(train_ratio=0.8, tr_eval_ratio=0.2)

    for i in range(30):
        # Train the model
        train_input_fn = gen_input_fn(train_data, train_labels, bs=32, ep=2, sh=True)
        action_classifier.train(
          input_fn=train_input_fn,
          steps=2000, #20000
          hooks=[logging_hook])

        # Evaluate training data
        eval_Tr_input_fn =gen_input_fn(train_data, train_labels, bs=32, ep=1, sh=True)
        eval_Tr_results = action_classifier.evaluate(input_fn=eval_Tr_input_fn)
        print("Evaluate Training data:", eval_Tr_results)
        # Evaluate evaluating data
        eval_input_fn =  gen_input_fn(eval_data, eval_labels, bs=32, ep=1, sh=True)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        print("Evaluate Evaluating data:",eval_results)
        # log
        acc_eval.append(eval_results['accuracy'])
        acc_train.append(eval_Tr_results['accuracy'])
        loss_train.append((eval_Tr_results['loss']))
        # Testing the model by filmed video.
        test_input_fn = gen_input_fn(test_data, test_labels, bs=1, ep=1, sh=True)
        test_results = action_classifier.evaluate(input_fn=test_input_fn)
        print("Evaluate Testing data:",test_results)
        acc_test.append(test_results['accuracy'])
    plot_acc_loss(acc_train,acc_eval,loss_train )


