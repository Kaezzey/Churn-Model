import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import os

from collections import Counter
from sklearn.metrics import accuracy_score
from collections import defaultdict

#read the data
df = pd.read_csv('churn.csv')

#clean the data
x = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

#convert the target variable to binary
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train
)

#convert class weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

#define the model path name
model_path = 'churn_model.keras'

if os.path.exists(model_path):

    #load the model
    model = load_model(model_path)

    #compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

else:
    #define the model
    model = Sequential()

    #add layers to the model
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(1, activation='sigmoid'))

    #compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #define early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    #fit the model
    model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weight_dict
    )

    #save the model
    model.save(model_path)

#evaulate the model

f1_0_scores = []
f1_1_scores = []
accuracies = []

# Initialize accumulators for all metrics
metrics_sum = defaultdict(float)
metrics_count = defaultdict(int)

for j in range(100):

    #split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    y_hat = model.predict(x_test)

    y_hat = [1 if i > 0.45 else 0 for i in y_hat]

    report = classification_report(y_test, y_hat, output_dict=True)

    #store the f1 scores and accuracy
    for label in ['0', '1']:
        
        for metric in ['precision', 'recall', 'f1-score', 'support']:

            key = f"{label}_{metric}"
            metrics_sum[key] += report[label][metric]
            metrics_count[key] += 1

    for avg in ['macro avg', 'weighted avg']:

        #store the average metrics
        for metric in ['precision', 'recall', 'f1-score', 'support']:

            key = f"{avg}_{metric}"
            metrics_sum[key] += report[avg][metric]
            metrics_count[key] += 1

    metrics_sum['accuracy'] += report['accuracy']

    metrics_count['accuracy'] += 1

#compute averages
avg_report = {}

#calculate the average for each metric
for key in metrics_sum:

    avg_report[key] = metrics_sum[key] / metrics_count[key]

#print header
print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
print("-" * 56)

#individual classes
for label in ['0', '1']:
    p = avg_report[f"{label}_precision"]
    r = avg_report[f"{label}_recall"]
    f1 = avg_report[f"{label}_f1-score"]
    s = avg_report[f"{label}_support"]
    print(f"{label:>12} {p:10.2f} {r:10.2f} {f1:10.2f} {int(s):10}")

#averages
for avg in ['macro avg', 'weighted avg']:
    p = avg_report[f"{avg}_precision"]
    r = avg_report[f"{avg}_recall"]
    f1 = avg_report[f"{avg}_f1-score"]
    s = avg_report[f"{avg}_support"]
    print(f"{avg:>12} {p:10.2f} {r:10.2f} {f1:10.2f} {int(s):10}")

#accuracy line
print(f"\n{'accuracy':<12} {avg_report['accuracy']:>32.2f}")

