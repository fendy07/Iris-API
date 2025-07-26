import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a function model
def train_model():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Classifier training
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Log the classification report
    print("Classification Report:")
    print(report)

    if accuracy > 0.9:
        joblib.dump(clf, 'model.joblib')
        return accuracy, True
    else:
        return accuracy, False