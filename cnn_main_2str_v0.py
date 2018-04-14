# version 2

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from readfunc_v2_dualstr import single_dataset_gen
from readfunc_v2_dualstr import examples
from readfunc_v2_dualstr import kfold_dataset_gen
from Visualize import plot_acc_bar
from Visualize import plot_acc_loss
from enum import Enum
from readfunc_v2_dualstr import FeatureType
tf.logging.set_verbosity(tf.logging.INFO)

class TopologyType(Enum):
    SingleCNN_RGB = 1
    SingleCNN_OF = 2
    SingleCNN_OF_MULTI = 3
    DurlCNN_RGB_OFs = 4
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
    TP = np.sum(np.diag(conf))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Precision:", precision)
    print("Recall:", recall)
    p = np.sum(precision) / 5.0
    r = np.sum(recall) / 5.0
    fscore = 2 * p * r / (p + r)
    print("F-Score:", fscore)

# CNN model function
def cnn_model_fn(features, labels, mode):
  # Input layer determined by topology
  flag_dual_tower = False
  if topologyType == TopologyType.SingleCNN_RGB:
      input_in1 = tf.reshape(features["x1"], [-1, 120, 120, 3]) #RGB
  elif topologyType == TopologyType.SingleCNN_OF:
      input_in1 = tf.reshape(features["x1"], [-1, 120, 120, 2]) #Single OF
  elif topologyType == TopologyType.SingleCNN_OF_MULTI:
      input_in1 = tf.reshape(features["x1"], [-1, 120, 120, 20]) #10 OFs
  elif topologyType == TopologyType.DurlCNN_RGB_OFs:
      input_in1 = tf.reshape(features["x1"], [-1, 120, 120, 3]) # RGB
      flag_dual_tower = True

  print(np.shape(input_in1))
  # 1st CNN Tower
  conv1_in1 = tf.layers.conv2d(input_in1, 32,[5, 5], padding="Same", activation=tf.nn.relu)
  print(np.shape(conv1_in1))
  pool1_in1 = tf.layers.max_pooling2d(inputs=conv1_in1, pool_size=[2, 2], strides=2)
  conv2_in1 = tf.layers.conv2d(pool1_in1, 64, [5, 5], padding="Same", activation=tf.nn.relu)
  print(np.shape(conv2_in1))
  pool2_in1 = tf.layers.max_pooling2d(inputs=conv2_in1, pool_size=[2, 2], strides=2)
  pool2_flat_in1 = tf.reshape(pool2_in1, [-1, 30 * 30 * 64 * 1])
  print(np.shape(pool2_flat_in1))
  dense_in1 = tf.layers.dense(inputs=pool2_flat_in1, units=128, activation=tf.nn.relu)
  print(np.shape(dense_in1))
  dropout_in1 = tf.layers.dropout(inputs=dense_in1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  logits_in1 = tf.layers.dense(inputs=dropout_in1, units=5)
  prob_in1 = tf.nn.softmax(logits_in1, name="softmax_tensor_in1")

  # 2nd CNN Tower
  if flag_dual_tower:
      #  Motion input (Optical Flow)
      input_in2 = tf.reshape(features["x2"], [-1, 120, 120, 20])
      conv1_in2 = tf.layers.conv2d(input_in2, 32,[5, 5], padding="Same", activation=tf.nn.relu)
      pool1_in2 = tf.layers.max_pooling2d(inputs=conv1_in2, pool_size=[2, 2], strides=2)
      conv2_in2 = tf.layers.conv2d(pool1_in2, 64, [5, 5], padding="Same", activation=tf.nn.relu)
      pool2_in2 = tf.layers.max_pooling2d(inputs=conv2_in2, pool_size=[2, 2], strides=2)
      pool2_flat_in2 = tf.reshape(pool2_in2, [-1, 30 * 30 * 64 * 1])
      dense_in2 = tf.layers.dense(inputs=pool2_flat_in2, units=128, activation=tf.nn.relu)
      dropout_in2 = tf.layers.dropout(inputs=dense_in2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
      logits_in2 = tf.layers.dense(inputs=dropout_in2, units=5)
      prob_in2 = tf.nn.softmax(logits_in2, name="softmax_tensor_in2")

  # Choose Single or Averaging probability
  if flag_dual_tower:
      # Late Fusion (averaging probability)
      score_avg = tf.divide(tf.add(prob_in1,prob_in2),2.0, "softmax_tensor")
  else:
      score_avg = tf.divide(tf.add(prob_in1,prob_in1),2.0, "softmax_tensor")

  # Classified by probability
  predictions = {
      "classes": tf.argmax(input=score_avg, axis=1,name="predication_classes"),
      "probabilities": score_avg
  }

  # Calculate the loss
  if flag_dual_tower:
      loss1 = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_in1)
      loss2 = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_in2)
      loss = tf.add(loss1,loss2)
  else:
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_in1)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07)
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

def gen_input_fn_wrapper(data,labels, bs=32,ep=2,sh=True, dualfeature=False):
    if dualfeature:
        f1, f2 = convert_two_features(data)
        x_features = {"x1": f1, "x2": f2}
        print("Feature1:" + str(np.shape(f1)))
        print("Feature2:" + str(np.shape(f2)))
    else:
        x_features = {"x1": data}
        print("Feature1:" + str(np.shape(data)))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_features,
        y=labels,
        batch_size=bs,
        num_epochs=ep,
        shuffle=sh)
    print("label:" + str(np.shape(labels)))
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

def evaluate_tr_ev_print_result(classifier, input_fn_tr, input_fn_eval):
    # For Training data
    eval_tr_results = classifier.evaluate(input_fn=input_fn_tr)
    print("Evaluate Training data result:", eval_tr_results)
    cal_p_r_fscore(eval_tr_results['conf_matrix'])

    # For Evaluating data
    eval_eval_results = classifier.evaluate(input_fn=input_fn_eval)
    print("Evaluate Evaluating data result:", eval_eval_results)
    cal_p_r_fscore(eval_eval_results['conf_matrix'])

    # Return accuracy
    acc_tr1 = eval_tr_results['accuracy']
    acc_eval1 = eval_eval_results['accuracy']
    return acc_tr1, acc_eval1
def evaluate_test_print_result(classifier, input_fn):
    eval_te_results = classifier.evaluate(input_fn=input_fn)
    print("Evaluate Testing data result:", eval_te_results)
    cal_p_r_fscore(eval_te_results['conf_matrix'])
    acc_te = eval_te_results['accuracy']
    return acc_te
def evaluate_film_print_result(classifier, input_fn):
    eval_film_results = classifier.evaluate(input_fn=input_fn)
    print("Evaluate Film data result:", eval_film_results)
    cal_p_r_fscore(eval_film_results['conf_matrix'])
    acc_te_film = eval_film_results['accuracy']
    return acc_te_film

def gen_dataset(tr_te_ratio1, tr_eval_ratio1, feature_type):
    print("Generate single dataset..")
    train1,eval1, test1, test_film = single_dataset_gen(tr_te_ratio1, tr_eval_ratio1, feature_type)
    train_data1 = train1.images
    train_labels1 = train1.labels
    eval_data1 = eval1.images
    eval_labels1 = eval1.labels
    test_data1 = test1.images
    test_labels1 =  test1.labels
    test_film_data1 = test_film.images
    test_film_labels1 = test_film.labels

    print("Get Training data:", np.shape(train_data1))
    print("Get Evaluating data:", np.shape(eval_data1))
    print("Get Testing data (Hold):", np.shape(test_data1))
    print("Get Testing Filmed data:", np.shape(test_film_data1))
    print(train_labels1)
    return train_data1, train_labels1, eval_data1, eval_labels1, test_data1, test_labels1,\
           test_film_data1, test_film_labels1
def convert_two_features(data):
    n = len(data)
    f1 = []
    f2 = []
    for i in range(n):
        f1.append(data[i].images1)
        f2.append(data[i].images2)
    f1 = np.asarray(f1,dtype=np.float32)
    f2 = np.asarray(f2,dtype=np.float32)
    return f1, f2

# --------------- Main Processing ----------------
#--------------- Configuration -------------------
# Choose topology
#topologyType = TopologyType.SingleCNN_RGB
topologyType = TopologyType.SingleCNN_OF
#topologyType = TopologyType.SingleCNN_OF_MULTI
#topologyType = TopologyType.DurlCNN_RGB_OFs

# One step action
Flag_Once_Generate_Dateset = True
Flag_Once_Training = True
Flag_Once_Evaluation = True
Flag_Once_Testing = True
Flag_Once_Testing_Film = True
# Combined action
Flag_Cross_Validation_Test = False
Flag_Reduce_Training_Test = False
Flag_Long_Training = False
# Basic Configuration
tr_te_ratio = 0.4 # Training data v.s. Testing data(Hold)
tr_ev_ratio = 0.1 # In Training data, Training v.s. Evaluate
model_dir1 = " "
model_index = 2
# Pre-setting
if topologyType == TopologyType.DurlCNN_RGB_OFs:
    Flag_Two_Stream = True
else:
    Flag_Two_Stream = False
# Choose features based on topology & Set model folder
if topologyType == TopologyType.SingleCNN_RGB:
    feature_type1 = FeatureType.RGB
    model_dir1 = "/tmp/cnn_model_RGB_" + str(model_index)
elif topologyType == TopologyType.SingleCNN_OF:
    feature_type1 = FeatureType.OPTICAL
    model_dir1 = "/tmp/cnn_model_OF_" + str(model_index)
elif topologyType == TopologyType.SingleCNN_OF_MULTI:
    feature_type1 = FeatureType.OPTICAL_MULTI
    model_dir1 = "/tmp/cnn_model_OFs_" + str(model_index)
elif topologyType == TopologyType.DurlCNN_RGB_OFs:
    feature_type1 = FeatureType.Dual
    model_dir1 = "/tmp/cnn_model_Dual_" + str(model_index)

# Create the Estimator
print("Estimator Creating...")
action_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir=model_dir1)

# Logging
print("Training Logging Setting...")
tensors_to_log = {"probabilities": "softmax_tensor",
                "classes": "predication_classes"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

# ------------------- One Step Action -----------------
# Load training, evaluating, test and filmed data with RGB,OF,OFs,Dual features.
if Flag_Once_Generate_Dateset:
    train_data, train_labels, eval_data, \
    eval_labels, test_data, test_labels, \
    test_film_data, test_film_labels= gen_dataset(tr_te_ratio,tr_ev_ratio, feature_type=feature_type1)

# Training the model
if Flag_Once_Training:
    train_input_fn = gen_input_fn_wrapper(train_data, train_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
    print("Training classifier...")
    action_classifier.train(
      input_fn=train_input_fn,
      steps=400, #20000
      hooks=[logging_hook])

# Evaluation of training data & evaluating data
if Flag_Once_Evaluation:
    eval_Tr_input_fn = gen_input_fn_wrapper(train_data, train_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
    eval_Ev_input_fn = gen_input_fn_wrapper(eval_data, eval_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
    acc_tr, acc_eval = evaluate_tr_ev_print_result(action_classifier,eval_Tr_input_fn, eval_Ev_input_fn )
# Test unseen test data
if Flag_Once_Testing:
    eval_Te_input_fn = gen_input_fn_wrapper(test_data, test_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
    acc_te = evaluate_test_print_result(action_classifier,eval_Te_input_fn)
# Test filmed data
if Flag_Once_Testing_Film:
    eval_Film_input_fn = gen_input_fn_wrapper(test_film_data, test_film_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
    acc_film = evaluate_film_print_result(action_classifier,eval_Film_input_fn)

# ------------------- Combined Actions -----------------

# Cross Validation
if Flag_Cross_Validation_Test:
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
        train_input_fn = gen_input_fn_wrapper(tr.images, tr.labels, bs=32, ep=1, sh=False, dualfeature=Flag_Two_Stream)
        # Train & Evaluate
        print("Model:", model_dir, "Training start")
        action_classifier.train(
            input_fn=train_input_fn, steps=800, hooks=[logging_hook])
        eval_input_fn = gen_input_fn_wrapper(ev.images, ev.labels, bs=32, ep=1, sh=False, dualfeature=Flag_Two_Stream)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        # Print Result
        print("Evaluate result:", eval_results)
        cal_p_r_fscore(eval_results['conf_matrix'])
        total_acc.append(eval_results['accuracy'])
        print(ev_ind)
        count += 1
    print("Average Acc:", sum(total_acc) / len(total_acc))
    plot_acc_bar(total_acc, 'Cross-Validation')

# Evaluate different size of training dataset
# Running four kinds of quantity of training examples
if Flag_Reduce_Training_Test:
    # four kinds of training usage ratio (e.g. 30% of whole dataset)
    train_ratios = [0.3,0.5,0.7,0.9]
    total_acc = []
    model_dir1 = "/tmp/cnn_model_reduce"
    count = 0
    print("Reduced training test starts...")
    for tr_ratio in train_ratios:
        # Generate new dataset by train_ratios
        train_data, train_labels, eval_data, \
        eval_labels, test_data, test_labels, \
        test_film_data, test_film_labels = gen_dataset(train_ratio=tr_ratio, tr_eval_ratio=0.3)
        # Create model
        model_dir = model_dir1 + '_' + str(count)
        action_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_dir)
        # Training
        print("Model:", model_dir, "Training start")
        train_input_fn = gen_input_fn_wrapper(train_data, train_labels, bs=16, ep=1, sh=True, dualfeature=Flag_Two_Stream)
        action_classifier.train(
          input_fn=train_input_fn,
          steps=500, #20000
          hooks=[logging_hook])
        # Evaluate
        eval_Tr_input_fn = gen_input_fn_wrapper(train_data, train_labels, bs=32, ep=1, sh=False, dualfeature=Flag_Two_Stream)
        eval_input_fn = gen_input_fn_wrapper(eval_data, eval_labels, bs=32, ep=1, sh=False, dualfeature=Flag_Two_Stream)
        acc_tr, acc_eval = evaluate_tr_ev_print_result(action_classifier, eval_Tr_input_fn, eval_input_fn)
        total_acc.append(acc_eval)
        count += 1
    print("Average Acc:", sum(total_acc) / len(total_acc))
    plot_acc_bar(total_acc, 'Four kinds of training ratio')

# Long 2500 steps Training
if Flag_Long_Training:
    acc_train = []
    acc_eval = []
    acc_test = []
    acc_test_film = []

    loss_train = []
    print("long long long training starts...")
    # Generate new dataset by train_ratios
    train_data, train_labels, eval_data, \
    eval_labels, test_data, test_labels, \
    test_film_data, test_film_labels = gen_dataset(train_ratio=0.8, tr_eval_ratio=0.2)

    for i in range(30):
        # Train the model
        train_input_fn = gen_input_fn_wrapper(train_data, train_labels, bs=32, ep=2, sh=True, dualfeature=Flag_Two_Stream)
        action_classifier.train(
          input_fn=train_input_fn,
          steps=2000, #20000
          hooks=[logging_hook])

        # Evaluate training data
        eval_Tr_input_fn =gen_input_fn_wrapper(train_data, train_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
        eval_Tr_results = action_classifier.evaluate(input_fn=eval_Tr_input_fn)
        print("Evaluate Training data:", eval_Tr_results)
        # Evaluate evaluating data
        eval_input_fn =  gen_input_fn_wrapper(eval_data, eval_labels, bs=32, ep=1, sh=True, dualfeature=Flag_Two_Stream)
        eval_results = action_classifier.evaluate(input_fn=eval_input_fn)
        print("Evaluate Evaluating data:",eval_results)
        # log
        acc_eval.append(eval_results['accuracy'])
        acc_train.append(eval_Tr_results['accuracy'])
        loss_train.append((eval_Tr_results['loss']))
        # Testing the model by filmed video.
        test_input_fn = gen_input_fn_wrapper(test_film_data, test_film_labels, bs=1, ep=1, sh=True, dualfeature=Flag_Two_Stream)
        test_results = action_classifier.evaluate(input_fn=test_input_fn)
        print("Evaluate Testing_Filmed data:",test_results)
        acc_test_film.append(test_results['accuracy'])
    plot_acc_loss(acc_train,acc_eval,loss_train )

# Debug
# The shape of each layer in CNN model
#for tmp in  action_classifier.get_variable_names():
#    print(np.shape(action_classifier.get_variable_value(name=tmp)))
