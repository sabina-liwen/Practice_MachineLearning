#Problem Statement -- exercise in the course of Machine Learning with Andraw Ng 
#Suppose that you are the administrator of a university department.
#You have historical data from previous applicants that you can use as a training set for logistic regression.
#For each training example, you have the applicant’s scores on two exams and the admissions decision.
#Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams.

#Request:
#Loading and visualizing the data
#You will start by loading the dataset for this task.
# The load_dataset() function shown below loads the data into variables X_train and y_train
#X_train contains exam scores on two exams for a student
#y_train is the admission decision
#y_train = 1 if the student was admitted
#y_train = 0 if the student was not admitted
#Both X_train and y_train are numpy arrays.


import numpy as np
import matplotlib.pyplot as plt

def load_dataset():

    np.random.seed(0)
    n_s = num_students = 100
    X1 = np.random.normal(70, 10, n_s )  # Exam 1
    X2 = np.random.normal(75, 12, n_s)  # Exam 2
    X_train = np.column_stack((X1, X2))

    y_train = ((0.3 * X1 + 0.7 * X2 ) > 75).astype(int)

    return X_train, y_train

def sigmoid(z):
    return 1/(1+np.exp(-z))
X_train, y_train = load_dataset()
n_s = num_students = 100
w = np.zeros(X_train.shape[1])
b = 0
LRate = 0.02
epoches = 1000
eps = 1e-8 

for epoch in range(epoches):
    z = np.dot(X_train,w) + b
    y_hat = sigmoid(z)
    Loss = -np.mean(y_train * np.log(y_hat + eps) + (1 - y_train) * np.log(1 - y_hat + eps))

    dw = np.dot(X_train.T, (y_hat - y_train)) / n_s 
    db = np.mean(y_hat - y_train)
    # renew w and b
    w -= LRate * dw
    b -= LRate * db
    if epoch % 100 == 0:
        print(f" the {epoch} time, Loss = {Loss:.2f}")

print("\n finsh!")
print("w =", w)
print("b =", b)

# loading visible data 
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], c='r', label='Not Admitted')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], c='g', label='Admitted')
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.title("Student Exam Scores and Admission")
plt.legend()
plt.grid(True)
plt.show()

# testing 
new_data = np.array([
    [85, 88],
    [60, 55],
    [72, 80]
])

z = np.dot(new_data, w) + b
probs = sigmoid(z)

for i, p in enumerate(probs):
    result = "admitted" if p >= 0.5 else "not admitted"
    print(f"Student {i+1}: prob = {round(p, 4)} → {result}")