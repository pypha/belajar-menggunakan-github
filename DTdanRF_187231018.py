#random forest
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report)
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import seaborn as sns

#Menampilkan semua data
dataframe = pd.read_excel('BlaBla.xlsx')
data = dataframe[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
print(" DATA AWAL ".center(47,"="))
print(data)
print("===============================================\n")

#Grouping
print(" GROUPING VARIABEL ".center(47,"="))
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print("data variabel".center(19,"="))
print(X)
print("data kelas".center(18,"="))
print(y)
print("===============================================\n")

#Training & testing
print(" SPLITTING DATA 20-80 ".center(75,"="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("instance variabel data training".center(75,"="))
print(X_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(X_test)
print("instance kelas data testing".center(75,"="))
print(y_test)
print("===========================================================================\n")

#Instance Prediksi
random_forest = RandomForestClassifier(random_state = 0)
random_forest.fit(X_train, y_train)

print("Instance prediksi random forest: ")
Y_pred = random_forest.predict(X_test)
print(Y_pred)
print("===========================================================================\n")

# Prediksi akurasi
accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("Akurasi: ", accuracy, "%\n")

# Classification report & confusion matrix Random Forest
print(" CLASSIFICATION REPORT RANDOM FOREST ".center(75,"="))
print(classification_report(y_test, Y_pred))

cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix:")
print(cm, "\n")

# Visualisasi Random Forest
feature_names_list = data.columns[:-1].tolist()
tree_to_plot = random_forest.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, filled=True, feature_names=feature_names_list, class_names=[str(i) for i in Counter(y_train).keys()])
plt.show()

# Klasifikasi / prediksi input Random Forest
print(" CONTOH INPUT ".center(75,"="))
A = int(input("Umur Pasien = "))
print("Isi jenis kelamin dengan 0 jika perempuan dan 1 jika laki-laki")
B = input("Jenis Kelamin Pasien = ")
print("Isi Y jika mengalami dan N jika tidak")
C = input("Apakah pasien mengalami C? = ")
D = input("Apakah pasien mengalami D? = ")
E = input("Apakah pasien mengalami E? = ")
F = input("Apakah pasien mengalami F? = ")
G = input("Apakah pasien mengalami G? = ")
H = input("Apakah pasien mengalami H? = ")
I = input("Apakah pasien mengalami I? = ")
J = input("Apakah pasien mengalami J? = ")
K = input("Apakah pasien mengalami K? = ")
L = input("Apakah pasien mengalami L? = ")
M = input("Apakah M? = ")


umur_k = 0
A_k = 0
B_k = 0


if A < 21:
    A_k = 1
if A > 20 and A < 31:
    A_k = 2
if A > 30 and A < 41:
    A_k = 3
if A > 40 and A < 51:
    A_k = 4
if A > 50:
    A_k = 5


print("Kode umur pasien adalah",A_k)


if B == "P":
    B_k = 1
else:
    B_k = 0


if C == "Y":
    C = 1
else:
    C = 0


if D == "Y":
    D = 1
else:
    D = 0


if E == "Y":
    E = 1
else:
    E = 0


if F == "Y":
    F = 1
else:
    F = 0


if G == "Y":
    G = 1
else:
    G = 0


if H == "Y":
    H = 1
else:
    H = 0


if I == "Y":
    I = 1
else:
    I = 0


if J == "Y":
    J = 1
else:
    J = 0


if K == "Y":
    K = 1
else:
    K = 0


if L == "Y":
    L = 1
else:
    L = 0


if M == "Y":
    M = 1
else:
    M = 0


Train = [A_k, B_k, C, D, E, F, G, H, I, J, K, L, M]
print(Train)


test = pd.DataFrame(Train).T
predtest = random_forest.predict(test)
print("HASIL PREDIKSI:", end=" ")
if predtest == 1:
    print("Pasien Positive")
else:
    print("Pasien Negative")

# Evaluasi model Random Forest
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
   
    print(f"\n{' METRIK EVALUASI '+model_name+' ':=^75}")
    print(f"Akurasi    : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Presisi    : {precision_score(y_test, y_pred):.2%}")
    print(f"Recall     : {recall_score(y_test, y_pred):.2%}")
    print(f"F1-Score   : {f1_score(y_test, y_pred):.2%}")
   
    if len(np.unique(y_test)) > 1:
        print(f"AUC-ROC    : {roc_auc_score(y_test, y_prob):.2%}")
   
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


evaluate_model(random_forest, X_test, y_test, "RANDOM FOREST")


# Plot ROC Curve untuk Random Forest
plt.figure(figsize=(8, 6))
if len(np.unique(y_test)) > 1:
    rf_probs = random_forest.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    plt.show()
else:
    print("ROC Curve tidak dapat dihitung karena masalah dalam data target")




#decision tree
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np

#Menampilkan semua data
dataframe = pd.read_excel('BlaBla.xlsx')
data = dataframe[['A','B','C','D','E','F','G','H','I','J','K','L','M','N']]
print(" DATA AWAL ".center(47,"="))
print(data)
print("===============================================\n")

#Grouping
print(" GROUPING VARIABEL ".center(47,"="))
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
print("data variabel".center(19,"="))
print(X)
print("data kelas".center(18,"="))
print(y)
print("===============================================\n")

#Training & testing
print(" SPLITTING DATA 20-80 ".center(75,"="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("instance variabel data training".center(75,"="))
print(X_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(X_test)
print("instance kelas data testing".center(75,"="))
print(y_test)
print("===========================================================================\n")


# Instance Prediksi
decision_tree = DecisionTreeClassifier(random_state = 0)
decision_tree.fit(X_train, y_train)

print("Instance prediksi decision tree: ")
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
print("===========================================================================\n")

# Prediksi akurasi
accuracy = round(accuracy_score(y_test, Y_pred) * 100, 2)
print("Akurasi: ", accuracy, "%\n")

# Classification report & confusion matrix Decision Tree
print(" CLASSIFICATION REPORT DECISION TREE ".center(75,"="))
print(classification_report(y_test, Y_pred))

cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix:")
print(cm, "\n")

# Visualisasi Decision Tree
feature_names_list = data.columns[:-1].tolist()
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=feature_names_list, class_names=[str(i) for i in Counter(y_train).keys()])
plt.show()

# Klasifikasi / prediksi input Decision Tree
print(" CONTOH INPUT ".center(75,"="))
A = int(input("Umur Pasien = "))
print("Isi jenis kelamin dengan 0 jika perempuan dan 1 jika laki-laki")
B = input("Jenis Kelamin Pasien = ")
print("Isi Y jika mengalami dan N jika tidak")
C = input("Apakah pasien mengalami C? = ")
D = input("Apakah pasien mengalami D? = ")
E = input("Apakah pasien mengalami E? = ")
F = input("Apakah pasien mengalami F? = ")
G = input("Apakah pasien mengalami G? = ")
H = input("Apakah pasien mengalami H? = ")
I = input("Apakah pasien mengalami I? = ")
J = input("Apakah pasien mengalami J? = ")
K = input("Apakah pasien mengalami K? = ")
L = input("Apakah pasien mengalami L? = ")
M = input("Apakah M? = ")


umur_k = 0
A_k = 0
B_k = 0


if A < 21:
    A_k = 1
if A > 20 and A < 31:
    A_k = 2
if A > 30 and A < 41:
    A_k = 3
if A > 40 and A < 51:
    A_k = 4
if A > 50:
    A_k = 5


print("Kode umur pasien adalah",A_k)


if B == "P":
    B_k = 1
else:
    B_k = 0


if C == "Y":
    C = 1
else:
    C = 0


if D == "Y":
    D = 1
else:
    D = 0


if E == "Y":
    E = 1
else:
    E = 0


if F == "Y":
    F = 1
else:
    F = 0


if G == "Y":
    G = 1
else:
    G = 0


if H == "Y":
    H = 1
else:
    H = 0


if I == "Y":
    I = 1
else:
    I = 0


if J == "Y":
    J = 1
else:
    J = 0


if K == "Y":
    K = 1
else:
    K = 0


if L == "Y":
    L = 1
else:
    L = 0


if M == "Y":
    M = 1
else:
    M = 0


Train = [A_k, B_k, C, D, E, F, G, H, I, J, K, L, M]
print(Train)


test = pd.DataFrame(Train).T
predtest = decision_tree.predict(test)
print("HASIL PREDIKSI:", end=" ")
if predtest == 1:
    print("Pasien Positive")
else:
    print("Pasien Negative")

# Evaluasi model Decision Tree
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
   
    print(f"\n{' METRIK EVALUASI '+model_name+' ':=^75}")
    print(f"Akurasi    : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Presisi    : {precision_score(y_test, y_pred):.2%}")
    print(f"Recall     : {recall_score(y_test, y_pred):.2%}")
    print(f"F1-Score   : {f1_score(y_test, y_pred):.2%}")
   
    if len(np.unique(y_test)) > 1:
        print(f"AUC-ROC    : {roc_auc_score(y_test, y_prob):.2%}")
   
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


evaluate_model(decision_tree, X_test, y_test, "DECISION TREE")


# Plot ROC Curve untuk Decision Tree
plt.figure(figsize=(8, 6))
if len(np.unique(y_test)) > 1:
    dt_probs = decision_tree.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, dt_probs):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Decision Tree')
    plt.legend()
    plt.show()
else:
    print("ROC Curve tidak dapat dihitung karena masalah dalam data target")
