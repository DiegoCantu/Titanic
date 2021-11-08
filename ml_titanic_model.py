import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    print(X_train)

    pickle.dump(model, open('titanic_model.pkl','wb'))

# =======================================================================================
if __name__ == '__main__':
    create_titanic_model()
