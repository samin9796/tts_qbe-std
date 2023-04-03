import pandas as pd
from sklearn.utils import resample
import os
import glob
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import numpy as np

# Get the current location
dir_path = os.path.dirname(os.path.realpath(__file__))

data_df = pd.read_csv("annotated_dataset_dutch.csv")
test_df = pd.read_csv(dir_path + "/data/processed/dtw/wav2vec2-dutch-large_transformer-L11_gos-kdl.csv")

# Split the data into three dataframes since we have three TTS models
test_split = np.array_split(test_df, 3)

OL_df = pd.DataFrame()
WE_df = pd.DataFrame()
HO_df = pd.DataFrame()

OL_df = test_split[0].reset_index(drop=True)
WE_df = test_split[1].reset_index(drop=True)
HO_df = test_split[2].reset_index(drop=True)

mixed_score = []

for i, row in data_df.iterrows():
    data_df.at[i,'query'] = 'TTS_OL_' + data_df['query'][i][3:]
data_df = data_df.sort_values(['query', 'reference']).reset_index(drop=True)
OL_df = OL_df.sort_values(['query', 'reference']).reset_index(drop=True)
WE_df = WE_df.sort_values(['query', 'reference']).reset_index(drop=True)
HO_df = HO_df.sort_values(['query', 'reference']).reset_index(drop=True)

print(OL_df.head(5))
print(WE_df.head(5))
print(HO_df.head(5))

for i, row in OL_df.iterrows():
    # human + TTS mixed
    mixed_score.append((OL_df['prediction'][i] + WE_df['prediction'][i] + HO_df['prediction'][i])/3)

OL_df['mixed_score'] = mixed_score
test_df = OL_df


# Separate majority and minority classes
df_majority = data_df[data_df.label == 0]
df_minority = data_df[data_df.label == 1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=5000,
                                 random_state = 0)
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=1000,
                                 random_state = 0)
# Combine minority class with downsampled majority class
df_up_down_sampled = pd.concat([df_majority_downsampled, df_minority_upsampled])

X = df_up_down_sampled.iloc[:, 3].values
y = df_up_down_sampled.iloc[:, 2].values
similarity_scores = test_df.iloc[:, 3].values
print(similarity_scores)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 0)

# Run all the algorithms

def algorithm(output_references):

  knn = KNeighborsClassifier(n_neighbors=2)
  knn.fit(X_train.reshape(-1, 1), y_train)
  rf = RandomForestClassifier(max_depth=30, random_state=0)
  rf.fit(X_train.reshape(-1, 1), y_train)
  dt = DecisionTreeClassifier(random_state=0)
  dt.fit(X_train.reshape(-1, 1), y_train)
  nb = MultinomialNB()
  nb.fit(X_train.reshape(-1, 1), y_train)
  svm_classifier = svm.SVC(kernel="poly", gamma=2)
  svm_classifier.fit(X_train.reshape(-1, 1), y_train)

  lst = [knn, rf, dt, nb, svm_classifier]
  columns = ["KNN", "Random Forest", "Decision Tree", "Naive Bayes", "SVM"]
  i = 0
  for classifier in lst:
    test_df[columns[i]] = get_predictions(classifier)
    i += 1
  df = test_df.sort_values('prediction', ascending=False)

  for index, row in df.iterrows():
    if row["KNN"] == 1 and row["Random Forest"] == 1 and row["Decision Tree"] == 1 and row["SVM"] == 1:
      output_references.append(row["reference"])
      #print(row["reference"])

  output_file = dir_path + "/output/1_output.csv"

  df.to_csv(output_file, index = False)
  return output_references

def get_predictions(classifier):
  y_pred = classifier.predict(similarity_scores.reshape(-1, 1))
  return y_pred

def list_the_files(output_references):
  # Get the current location
  dir_path = os.path.dirname(os.path.realpath(__file__))
  dir_references = glob.glob(dir_path + "/data/raw/datasets/gos-kdl/references/*.wav")

  dst = dir_path + "/output"
  for src in dir_references:
      filename = src.split('/')[-1][:-4]
      if filename in output_references:
        #print(filename)
        shutil.copy(src, dst)
      


# Run the algorithms

output_references = []
output_references = algorithm(output_references)

list_the_files(output_references)
print('DONE')