# DECISION TREE CLASSIFICATION

# REQUIRED LIBRARIES
import pandas as pd

# IMPORTING DATA
data = pd.read_csv("Data..csv")
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

# SPLITING THE DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# TRAINING THE MODEL
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train,Y_train)

# TESTING THE MODEL
Y_pred = model.predict(X_test)

# EVALUATING THE MODEL WITH CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
CF = confusion_matrix(Y_test, Y_pred)
AS = accuracy_score(Y_test, Y_pred)
print("CONFUSION MATRIX")
print(CF)
print("..............")
print("RESULTS AFTER EVALUATION USING ACCURACY THROUGH CONFUSION MATRIX")
print("Accuracy:", AS*100)

