import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import pickle

# =======================================================================================
def create_titanic_model():
    titanic = pd.read_csv('train.csv')
    ports = pd.get_dummies(titanic.Embarked , prefix='Embarked')
    titanic = titanic.join(ports)
    titanic.drop(['Embarked'], axis=1, inplace=True)
    titanic.Sex = titanic.Sex.map({'male':0, 'female':1})
    y = titanic.Survived.copy()
    X = titanic.drop(['Survived'], axis=1)
    X.drop(['Cabin'], axis=1, inplace=True)
    X.drop(['Ticket'], axis=1, inplace=True)
    X.drop(['Name'], axis=1, inplace=True)
    X.drop(['PassengerId'], axis=1, inplace=True)
    X.Age.fillna(X.Age.mean(), inplace=True)
   
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
    model = LogisticRegression(solver='liblinear', multi_class='ovr')

    model.fit(X_train, y_train)

    # Predecimos las probabilidades
    probabilidad = model.predict_proba(X_valid)
    # Sacamos los valores
    prediccion = model.predict(X_valid)

    # get_confusion_matrix(prediccion)
    get_curva_roc(y_valid,probabilidad)
    get_precision_sensibilidad(y_valid,prediccion)

    pickle.dump(model, open('titanic_model.pkl','wb'))

# =======================================================================================
def get_curva_roc(y_valid,lr_probs):
    #Generamos un clasificador sin entrenar , que asignará 0 a todo
    ns_probs = [0 for _ in range(len(y_valid))]
    # Predecimos las probabilidades
    #Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
    lr_probs = lr_probs[:, 1]
    # Calculamos las curvas ROC
    ns_fpr, ns_tpr, _ = roc_curve(y_valid, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_valid, lr_probs)
    # Pintamos las curvas ROC
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Regresión Logística')
    # Etiquetas de los ejes
    pyplot.xlabel('Tasa de Falsos Positivos')
    pyplot.ylabel('Tasa de Verdaderos Positivos')
    pyplot.legend()
    pyplot.show()

# =======================================================================================
def get_precision_sensibilidad(y_valid,prediccion):
    
    lr_precision, lr_recall, _ = precision_recall_curve(y_valid, prediccion)

    # Resumimos
    print('Regresión Logística: f1=%.3f auc=%.3f' % (get_score_f1(y_valid,prediccion), get_auc(lr_recall,lr_precision)))
    # plot the precision-recall curves
    no_skill = len(y_valid[y_valid==1]) / len(y_valid)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Sin entrenar')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Regresión Logística')
    #Etiquetas de ejes
    pyplot.xlabel('Sensibilidad')
    pyplot.ylabel('Precisión')

    pyplot.legend()
    pyplot.show()

# =======================================================================================
def get_score_f1(y_valid,yhat):
    return f1_score(y_valid, yhat)

# =======================================================================================
def get_auc(recall,precision):
    return auc(recall, precision)

# =======================================================================================
def get_simple_confusion_matrix(predictions):
    """
    """
    y_true = []
    y_pred = []
    labels = set()
    for prediction in predictions:
        y_true.append(prediction.observed_class)
        y_pred.append(prediction.predicted_class)
        labels.add(prediction.observed_class)
        labels.add(prediction.predicted_class)
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return (labels, matrix)
# =======================================================================================
def get_confusion_matrix(predictions):
    """ Esta función convierte las predicciones que ya tienen calificación en la matriz
        de confusión de desempeño del modelo.
        :param predictions: La lista de predicciones calificadas
        :return: Un diccionario con la matriz de confusión del modelo
    """

    labels, matrix = get_simple_confusion_matrix(predictions)
    result_matrix = {}

    for actual_label, row in zip(labels, matrix):
        result_matrix[actual_label] = {}
        for label, count in zip(labels, row):
            result_matrix[actual_label][label] = int(count)
    
    return result_matrix

# =======================================================================================
if __name__ == '__main__':
    create_titanic_model()
