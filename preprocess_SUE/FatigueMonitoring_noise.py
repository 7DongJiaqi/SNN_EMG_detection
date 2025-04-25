from EMG_training import evaluateClassifier
from EMG_training import featureExtraction
from EMG_training import Classify


import os
import glob
import numpy
from pyAudioAnalysis import audioTrainTest as aT
from scipy.signal import medfilt
import Classification as clf
from copy import copy

import csv

import numpy as np
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(42)

np.random.seed(42)

tf.random.set_seed(42)
import tensorflow as tf

def add_noise_tf(data, noise_type=1, epsilon=0.1):
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    if noise_type == 1:
        noise = tf.random.uniform(shape=tf.shape(data_tensor), minval=-epsilon, maxval=epsilon)
    elif noise_type == 2:
        noise = tf.random.normal(shape=tf.shape(data_tensor), stddev=epsilon)
    elif noise_type == 3:
        noise = tf.random.poisson(shape=tf.shape(data_tensor), lam=1) * epsilon
    else:
        raise ValueError("Unsupported noise type")

    noisy_data = data_tensor + noise
    return noisy_data.numpy()




def computeEvalMetrics(CM, filename='results', save=False):
    CM = CM + 0.0000000010
    Rec = numpy.zeros((CM.shape[0],))
    Pre = numpy.zeros((CM.shape[0],))
    for ci in range(CM.shape[0]):
        Rec[ci] = CM[ci, ci] / numpy.sum(CM[ci, :])
        Pre[ci] = CM[ci, ci] / numpy.sum(CM[:, ci])
    F1 = 2 * Rec * Pre / (Rec + Pre)

    avg_pre = numpy.mean(Pre)
    avg_rec = numpy.mean(Rec)
    avg_f1 = numpy.mean(F1)

    print('Total CM')
    print(CM)
    print('Pre:', Pre, '- AVG Pre:', avg_pre)
    print('Rec:', Rec, '- AVG Rec:', avg_rec)
    print('F1:', F1, '- AVG F1:', avg_f1)

    if save:
        # Save .npz file
        results = {
            'Confusion_Matrix': CM,
            'Precision': Pre,
            'Recall': Rec,
            'F1_Score': F1,
            'AVG_Precision': avg_pre,
            'AVG_Recall': avg_rec,
            'AVG_F1_Score': avg_f1
        }
        numpy.savez(filename + "_results.npz", **results)

        # Save .csv file
        csv_filename = filename + "_metrics.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Values'])
            writer.writerow(['Confusion Matrix', CM.flatten()])
            writer.writerow(['Precision', Pre])
            writer.writerow(['Recall', Rec])
            writer.writerow(['F1 Score', F1])
            writer.writerow(['AVG Precision', avg_pre])
            writer.writerow(['AVG Recall', avg_rec])
            writer.writerow(['AVG F1 Score', avg_f1])



def postProcessing_2(predictions, filter=True):
    if filter:
        predictions = list(medfilt(predictions, 3))
    w = 3  # window length
    step = 1  # window step
    base_thresh = 0.6  # base confidence threshold
    alpha = 0.2  # smoothing factor for EMA
    ema_thresh = base_thresh  # initial EMA threshold is the base threshold

    # prev3 = None
    prev2 = None  # history window 1
    prev1 = None  # history window 2
    N = 0
    N1 = w
    count = 0
    complete = False  # if post processing didn't capture fatigue return predictions full of zeros ie. NO-FATIGUE
    fast_thresh = 0.8
    while True:
        count += 1
        if N1 > len(predictions):
            N1 = len(predictions) + 1
            complete = True
        x = predictions[N:N1]

        if x.count(1) / float(w) >= fast_thresh:
            index = count - 1
            post_predictions = [0] * index + [1] * (len(predictions) - index)
            print("Triggered fast channel")
            break

        current_ratio = x.count(1) / float(w)
        ema_thresh = alpha * current_ratio + (1 - alpha) * ema_thresh  # Update EMA threshold

        if prev2 != None:
            if x.count(1) / float(w) >= ema_thresh and prev1.count(1) / float(w) >= ema_thresh and prev2.count(
                    1) / float(
                    w) >= ema_thresh:
                index = count - 1
                post_predictions = [0] * index + [1] * len(predictions[index:])
                break
        prev3 = copy(prev2)
        prev2 = copy(prev1)
        prev1 = copy(x)
        N += step
        N1 += step

        if complete:
            post_predictions = [0] * len(predictions)
            break

    return post_predictions


def postProcessing(predictions, filter=True):
    if filter:
        predictions = list(medfilt(predictions, 3))

    w = 3  # window length
    step = 1  # int(w/3) #window step
    thresh = 0.6  # confidence threshold

    prev3 = None
    prev2 = None  # history window 1
    prev1 = None  # history window 2
    N = 0
    N1 = w
    count = 0
    complete = False  # if post processing didnt capture fatigue return predictions full of zeros ie. NO-FATIGUE
    while True:
        count += 1
        if N1 > len(predictions):
            N1 = len(predictions) + 1
            complete = True
        x = predictions[N:N1]

        if prev2 != None:
            if x.count(1) / float(w) >= thresh and prev1.count(1) / float(w) >= thresh and prev2.count(1) / float(
                    w) >= thresh:
                post_predictions = [0] * count + [1] * len(predictions[count:])
                break

        prev3 = copy(prev2)
        prev2 = copy(prev1)
        prev1 = copy(x)
        N += step
        N1 += step

        if complete:
            print('Complete postProcessing')
            post_predictions = [0] * len(predictions)
            break
    return post_predictions



def SingleUserEvaluation(dirName, classifier, c_param):
    fileList = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    user_ids = sorted(list(set(map(lambda x: x.split('/')[-1].split('_')[0], fileList))))
    CM_avg = numpy.zeros((2, 2))
    CM_avg_post = numpy.zeros((2, 2))
    CM_avg_post_2 = numpy.zeros((2, 2))

    for uid, user in enumerate(user_ids):
        CM = numpy.zeros((2, 2))
        CM_post = numpy.zeros((2, 2))
        CM_post_2 = numpy.zeros((2, 2))
        user_files = filter(lambda x: x.split('/')[-1].split('_')[0] == user, fileList)
        for file_test in user_files:
            train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(user_files, file_test, '')
            CM_user, CM_user_post, CM_user_post_2, predictions, post_predictions, post_predictions_2= GroupClassification(
                [train_data_NF, train_data_F], [test_data_NF, test_data_F], classifier, c_param, test_ids, dirName)
            CM += CM_user
            CM_post += CM_user_post
            CM_post_2 += CM_user_post_2
        CM_avg += CM

        CM_avg_post += CM_post
        CM_avg_post_2 += CM_post_2

        PrintResults('-', "TOTAL RESULTS ACROSS FILES OF USER " + user, CM, CM_post)
        PrintResults('-2', "TOTAL RESULTS ACROSS FILES OF USER " + user, CM, CM_post_2)
    PrintResults('+', "TOTAL RESULTS ACROSS ALL USERS", CM_avg, CM_avg_post)
    PrintResults('+2', "TOTAL RESULTS ACROSS ALL USERS", CM_avg, CM_avg_post_2)



def GroupClassification(train, test, classifier, param, test_ids, eval_source):
    CM = numpy.zeros((2, 2))
    CM_post = numpy.zeros((2, 2))
    CM_post_2 = numpy.zeros((2, 2))
    # from lists to matrices
    trNF = numpy.concatenate(train[0])
    trF = numpy.concatenate(train[1])
    print("trNF", len(trNF))
    # print(trNF)
    print("trF", len(trF))
    # normalize train features - 0mean -1std
    features_norm, MEAN, STD = clf.normalizeFeatures([trNF, trF])
    # print("features_norm",features_norm,len(features_norm))
    # train the classifier
    model= Classify(classifier, features_norm, param)
    # TEST
    # print(len(test[1]),"==========================test")
    for recording in range(len(test[0])):
        predictions = []
        probs = []
        print("TEST NF", test[0][recording].shape[0], "TEST F", test[1][recording].shape[0])
        test_labels = [0] * test[0][recording].shape[0] + [1] * test[1][recording].shape[0]
        test_recording_fVs = numpy.concatenate((test[0][recording], test[1][recording]))
        for i in range(test_recording_fVs.shape[0]):
            fV = test_recording_fVs[i, :]

            fV = (fV - MEAN) / STD  # fV.shape = (132,)
            [Result, P] = clf.classifierWrapper(model, classifier, fV)  # classification
            probs.append(numpy.max(P))
            predictions.append(Result)

        for idx, gtlabel in enumerate(test_labels):
            CM[int(gtlabel), int(predictions[idx])] += 1

        post_predictions = postProcessing(predictions)
        for idx, gtlabel in enumerate(test_labels):
            CM_post[int(gtlabel), int(post_predictions[idx])] += 1

        post_predictions_2 = postProcessing_2(predictions)
        for idx, gtlabel in enumerate(test_labels):
            CM_post_2[int(gtlabel), int(post_predictions_2[idx])] += 1

        # CompareToInitialStudy(post_predictions,test_ids[recording],eval_source)
    return CM, CM_post, CM_post_2, predictions, post_predictions, post_predictions_2


def DataCollect(filelist, test_id, evaluation_type):
    test_data_F = []
    test_data_NF = []
    train_data_F = []
    train_data_NF = []
    test_ids = []

    test_ids.append(test_id)
    with numpy.load(test_id) as data:
        test_data_NF.append(data[list(data.keys())[0]])
    data.close()
    with numpy.load(test_id.replace('NF', 'F')) as data:
        test_data_F.append(data[list(data.keys())[0]])
    data.close()

    for file in filelist:
        with numpy.load(file) as data:
            train_data_NF.append(data[list(data.keys())[0]])
        data.close()
        with numpy.load(file.replace('NF', 'F')) as data:
            train_data_F.append(data[list(data.keys())[0]])
        data.close()
    test_data_NF = [add_noise_tf(sample, noise_type=1, epsilon=0.3) for sample in test_data_NF]
    test_data_F = [add_noise_tf(sample, noise_type=1, epsilon=0.3) for sample in test_data_F]

    print(len(train_data_F), len(train_data_NF), len(test_data_F), len(test_data_NF), len(test_ids), test_ids)
    return train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids


def PrintResults(s, str, CM, CM_post):
    '''
    Prints results before and after post-processing
    '''
    print(s * 20)
    print(str)
    print("NO-POSTPROCESSING")
    computeEvalMetrics(CM)
    print("POSTPROCESSING")
    computeEvalMetrics(CM_post)




if __name__ == '__main__':
    '''
    NOTES
    median filter -11
    fixed post processing
    '''

    SingleUserEvaluation("../data/Study1_medfilt11_EMG", 'gradientboosting', 1000)
    SingleUserEvaluation("../data/Study1_medfilt11_EMG",'svm_rbf',2)
    SingleUserEvaluation("../data/Study1_medfilt11_EMG",'randomforest',500)
    SingleUserEvaluation("../data/Study1_medfilt11_EMG",'extratrees',500)
    SingleUserEvaluation("../data/Study1_medfilt11_EMG",'svm',100)
    SingleUserEvaluation("../data/Study1_medfilt11_EMG",'knn',5)


