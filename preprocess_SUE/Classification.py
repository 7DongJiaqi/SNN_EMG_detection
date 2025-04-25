import numpy
import sklearn
from scipy.spatial import distance


class kNN:
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k

    def classify(self, test_sample):
        n_classes = numpy.unique(self.Y).shape[0]
        y_dist = (distance.cdist(self.X,
                                 test_sample.reshape(1,
                                                     test_sample.shape[0]),
                                 'euclidean')).T
        i_sort = numpy.argsort(y_dist)
        P = numpy.zeros((n_classes,))
        for i in range(n_classes):
            P[i] = numpy.nonzero(self.Y[i_sort[0][0:self.k]] == i)[0].shape[0] / float(self.k)
        return (numpy.argmax(P), P)







def calculate_flops(model_type, n_samples, n_features, n_estimators=None, max_depth=None, n_support_vectors=None):
    if model_type in ['gradientboosting', 'randomforest', 'extratrees']:
        if not n_estimators:
            n_estimators = {'gradientboosting': 1000, 'randomforest': 500, 'extratrees': 500}[model_type]
        if not max_depth:
            max_depth = 3
        max_features = int(numpy.sqrt(n_features)) if model_type != 'gradientboosting' else n_features
        nodes_per_tree = 2 ** (max_depth + 1) - 1
        return n_estimators * nodes_per_tree * n_samples * max_features
    elif model_type == 'svm':
        if n_support_vectors is None:
            n_support_vectors = n_samples
        return n_support_vectors * n_features
    elif model_type == 'svm_rbf':
        if n_support_vectors is None:
            n_support_vectors = n_samples
        return n_support_vectors * n_support_vectors * n_features
    elif model_type == 'knn':
        return n_samples * n_samples * n_features*2
    else:
        return 0


def trainSVM(features, Cparam):
    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)
    svm.fit(X,Y)
    flops = calculate_flops('svm', X.shape[0], X.shape[1])
    return svm,flops



def trainKNN(features, K):
    [Xt, Yt] = listOfFeatures2Matrix(features)
    knn = kNN(Xt, Yt, K)
    flops = calculate_flops('knn', Xt.shape[0], Xt.shape[1])
    return knn,flops


def trainRandomForest(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)
    flops = calculate_flops('randomforest', X.shape[0], X.shape[1], n_estimators=n_estimators)
    return rf,flops


def trainGradientBoosting(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.GradientBoostingClassifier(n_estimators = n_estimators)
    rf.fit(X,Y)
    flops = calculate_flops('gradientboosting', X.shape[0], X.shape[1], n_estimators=n_estimators)
    return rf,flops


def trainExtraTrees(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    et = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimators)
    et.fit(X,Y)
    flops = calculate_flops('extratrees', X.shape[0], X.shape[1], n_estimators=n_estimators)
    return et,flops



def trainSVM_RBF(features, Cparam):
    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'rbf',  probability = True)
    svm.fit(X,Y)
    flops = calculate_flops('svm_rbf', X.shape[0], X.shape[1])
    return svm,flops


def listOfFeatures2Matrix(features):
    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)


def normalizeFeatures(features):
    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)



def classifierWrapper(classifier, classifier_type, test_sample):
    R = -1
    P = -1
    if classifier_type == "knn":
        [R, P] = classifier.classify(test_sample)
    elif classifier_type == "svm" or \
                    classifier_type == "randomforest" or \
                    classifier_type == "gradientboosting" or \
                    classifier_type == "extratrees" or \
                    classifier_type == "svm_rbf" or\
                    classifier_type == "gmm":
        R = classifier.predict(test_sample.reshape(1,-1))[0]
        P = classifier.predict_proba(test_sample.reshape(1,-1))[0]
    return [R, P]






