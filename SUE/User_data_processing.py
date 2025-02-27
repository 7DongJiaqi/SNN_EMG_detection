import os
import glob
import numpy
from User_MLP_quantize import *
# from try1 import *
from copy import copy
from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix


def computeEvalMetrics(CM,savepath, filename='results', save=False):
    '''
    Compute Precision, Recall & F1 given a Confusion Matrix
    :param CM: confusion matrix
    :param filename: name of output file - only if save is true
    :param save: if results will be stored
    :return: stores confusion matrix if save==True
    '''
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
    with open(f'{savepath}/log.txt', 'a') as log_file:

        log_file.write(f"AVG F1: {avg_f1}\n")
        log_file.write(f"----------------------\n")

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
        # print("filename ", filename)
        with open(filename + ".txt", 'a') as f:
            # f.write("results:\n")  # Optional: Add a separator or title for each run
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def postProcessing(predictions,filter=True):
    '''
    Performs post-processing of the raw classifier predictions based on a history of 2 previous windows of length w
    :param predictions: list of predicted labels
    :return: updated predictions after post processing
    '''

    if filter:
        predictions = list(medfilt(predictions,3))

    w = 3 #window length
    step = 1#int(w/3) #window step
    thresh = 0.6 #confidence threshold

    prev3 = None
    prev2 = None #history window 1
    prev1 = None #history window 2
    N=0
    N1 = w
    count = 0
    complete = False #if post processing didnt capture fatigue return predictions full of zeros ie. NO-FATIGUE
    while True:
        count += 1
        if N1 > len(predictions):
            N1 = len(predictions)+1
            complete = True
        x = predictions[N:N1]

        if prev2 != None:
            if x.count(1)/float(w)>=thresh and prev1.count(1)/float(w)>=thresh and prev2.count(1)/float(w)>=thresh :
                post_predictions = [0]*count+[1]*len(predictions[count:])
                break

        prev3 = copy(prev2)
        prev2 = copy(prev1)
        prev1 = copy(x)
        N += step
        N1 += step

        if complete:
            print ('Complete postProcessing')
            post_predictions = [0]*len(predictions)
            break
    return post_predictions

def postProcessing_2(predictions, filter=True):
    print(predictions)
    if filter:
        predictions = list(medfilt(predictions, 3))
        print('after filter', predictions)

    w = 3  # window length
    step = 1  # window step
    base_thresh = 0.6  # base confidence threshold
    alpha = 0.2 # smoothing factor for EMA
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
        print('x', N, N1)

        if x.count(1) / float(w) >= fast_thresh:
            # print(count)
            index = count-1
            post_predictions = [0] * index + [1] * (len(predictions) - index)
            print("Triggered fast channel")
            break

        current_ratio = x.count(1) / float(w)
        ema_thresh = alpha * current_ratio + (1 - alpha) * ema_thresh  # Update EMA threshold

        if prev2 != None:
            if x.count(1) / float(w) >= ema_thresh and prev1.count(1) / float(w) >= ema_thresh and prev2.count(1) / float(
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
            print('Complete postProcessing222')
            post_predictions = [0] * len(predictions)
            break

    # print("post_predictions_2",post_predictions)
    return post_predictions


def PrintResults(s,str,CM,CM_post, CM_post2, savepath, filename='snn/qann_99'):
    '''
    Prints results before and after post-processing
    '''
    print  (s*20)
    print (str)
    print( "NO-POSTPROCESSING")
    with open(filename + ".txt", 'a') as f:
        f.write("NO-POSTPROCESSING:\n")  # Optional: Add a separator or title for each run
    computeEvalMetrics(CM, savepath, filename, save=True)

    print ("POSTPROCESSING")
    with open(filename + ".txt", 'a') as f:
        f.write("POSTPROCESSING-1:\n")
    computeEvalMetrics(CM_post, savepath, filename, save=True)

    print ("POSTPROCESSING-2")
    with open(filename + ".txt", 'a') as f:
        f.write("POSTPROCESSING-2:\n")
    computeEvalMetrics(CM_post2, savepath, filename, save=True)



def GroupClassification(train,test,classifier,num_epochs,batch_size,lr,user, loadpath, savepath):
    CM = numpy.zeros((2, 2))
    CM_post = numpy.zeros((2, 2))
    CM_post_2 = numpy.zeros((2, 2))
    # from lists to matrices
    trNF = numpy.concatenate(train[0])
    trF = numpy.concatenate(train[1])
    # print("trNF",len(trNF))
    # print(trNF)
    # print("trF",len(trF))
    # print(trF)
    # normalize train features - 0mean -1std
    features_norm, MEAN, STD = normalizeFeatures([trNF, trF])
    [X_train, y_train] = listOfFeatures2Matrix(features_norm)

    #test
    for recording in range(len(test[0])):

        y_test = [0] * test[0][recording].shape[0] + [1] * test[1][recording].shape[0]
        X_test = numpy.concatenate((test[0][recording], test[1][recording]))
        X_test = (X_test - MEAN) / STD

    trainer = ModelTrainer(model_type=classifier, num_epochs=num_epochs, batch_size=batch_size, lr=lr, user=user, loadpath=loadpath, savepath=savepath)
    predictions, probs, CM = trainer.run(X_train, y_train, X_test, y_test)

    post_predictions = postProcessing(predictions)
    post2_predictions = postProcessing_2(predictions)
    # print(predictions)
    print("post", post_predictions)
    print('gt', y_test)
    CM_post = confusion_matrix(y_test, post_predictions)
    CM_post_2 = confusion_matrix(y_test, post2_predictions)
    return CM,CM_post, CM_post_2 ,predictions,post_predictions, post2_predictions





def DataCollect(filelist,test_id):
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

    # print(len(train_data_F),len(train_data_NF),len(test_data_F),len(test_data_NF),len(test_ids),test_ids)
    return train_data_F,train_data_NF, test_data_F,test_data_NF,test_ids





def SingleUserEvaluation(dirName,classifier,num_epochs,batch_size,lr,loadpath=None, savepath='model/cq'):
    fileList  = sorted(glob.glob(os.path.join(dirName, "*NF.npz")))
    # print(fileList)
    user_ids =  sorted(list(set(map(lambda x:x.split('/')[-1].split('_')[0],fileList))))
    CM_avg= numpy.zeros((2, 2))
    CM_avg_post = numpy.zeros((2, 2))
    CM_avg_post2 = numpy.zeros((2, 2))
    for uid,user in enumerate(user_ids):
        print(" ")
        print ("Test on user:", user)
        with open("snn/qann_99.txt", 'a') as f:
            f.write(user+":\n")
        CM = numpy.zeros((2, 2))
        CM_post = numpy.zeros((2, 2))
        CM_post2 = numpy.zeros((2, 2))
        user_files = filter(lambda x:x.split('/')[-1].split('_')[0] == user,fileList)

        for file_test in user_files:
            # print("file_test",file_test)
            train_data_F, train_data_NF, test_data_F, test_data_NF, test_ids = DataCollect(user_files,file_test)
            CM_user, CM_user_post, CM_user_post2, predictions, post_predictions, post2_predictions = GroupClassification([train_data_NF, train_data_F], [test_data_NF, test_data_F], classifier,num_epochs,batch_size,lr,user, loadpath, savepath)
            CM += CM_user
            CM_post += CM_user_post
            CM_post2 += CM_user_post2
            
        CM_avg += CM
        CM_avg_post +=CM_post
        CM_avg_post2 +=CM_post2
        PrintResults('-',"TOTAL RESULTS ACROSS FILES OF USER "+user,CM,CM_post, CM_post2, savepath)
    PrintResults('+',"TOTAL RESULTS ACROSS ALL USERS",CM_avg,CM_avg_post, CM_avg_post2, savepath)

