import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from keras import backend as K

def model_predict_USL(X_scaled_true, clf, X_test):
    clf.fit(X_scaled_true)
    yhat = clf.predict(X_test)
    y_pred=pd.Series(yhat).replace([-1,1],[1,0])
    return y_pred


def report_USL(y_pred, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("CLASSIFICATION REPORT")
    print(class_report)
    print("CONFUSION MATRIX") 
    print(conf_matrix)
    return conf_matrix
    
def model_predict_SL(gs, X_train, y_train, X_test):
    yhat = pd.Series(y_train).replace([1,0],[-1,1])
    gs.fit(X_train, yhat)
    y_pred = gs.predict(X_test)
    return y_pred, gs

def report_SL(y_pred, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    yhat = pd.Series(y_pred).replace([-1,1], [1,0])
    class_report = classification_report(y_test, yhat)
    conf_matrix = confusion_matrix(y_test, yhat)
    print("CLASSIFICATION REPORT")
    print(class_report)
    print("CONFUSION MATRIX")   
    print(conf_matrix)
    return conf_matrix
    
    
    
def plot_gridsearch_cv(results, estimator, scoring, x_min, x_max, y_min, y_max):
    
    # print GridSearch cross-validation for parameters
    
    plt.figure(figsize=(10,8))
    plt.title("GridSearchCV for "+estimator, fontsize=24)

    plt.xlabel(estimator)
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    pad = 0.005
    X_axis = np.array(results["param_"+estimator].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['b', 'k']):
        sample_score_mean = results['mean_test_%s' % (scorer)]
        sample_score_std = results['std_test_%s' % (scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1, color=color)
        ax.plot(X_axis, sample_score_mean, '-', color=color,
            alpha=1,
            label="%s (test)" % (scorer))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score+pad))

    plt.legend(loc="best")
    plt.grid('off')
    plt.tight_layout()
        
    plt.show()
    
    
def recall(y_true, y_pred):
    """Recall metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1_factor(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

def recall_cm(class_report):
    return class_report[1][1]/(class_report[1][0]+class_report[1][1]+K.epsilon())

def precision_cm(class_report):
    return class_report[1][1]/(class_report[0][1]+class_report[1][1]+K.epsilon())

def precision_cm0(class_report):
    return class_report[0][0]/(class_report[1][0]+class_report[0][0]+K.epsilon())

def f1_factor_cm(class_report):
    prec = precision_cm(class_report)
    rec = recall_cm(class_report)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))
    

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
