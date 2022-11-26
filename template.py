#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(dataset_path):
	data_load = pd.read_csv(dataset_path)
	return data_load
	#To-Do: Implement this function 주소입력을 받아와서 pandas를 이용해 데이터셋을 로드한다

def dataset_stat(dataset_df):
	data_stat = dataset_df.data.number_of_features, dataset_df.data.number_of_target[0], dataset_de.data.number_of_target[1]
	return data_stat
	#To-Do: Implement this function featuers의 수, class0인 target0의 수, class1인 target1의 수를 반환

def split_dataset(dataset_df, testset_size):
	data_split=train_test_split(dataset_df.data, dataset_df.target, testset_size)
	return data_split
	#To-Do: Implement this function train 데이터, test 데이터,train 레이블, test레이블로 사이즈를 받아서 분할한다

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	data_decision = accuracy_score(x_test,y_test), precision_score(x_test, y_test), recall_score(x_test, y_test)
	return data_decision
	#To-Do: Implement this function 스플리트로 나눈 트레인을 정확성 정밀도 리콜 순으로 반환했다

def random_forest_train_test(x_train, x_test, y_train, y_test):
	pipe=make_pipeline(StandardScaler(),RandomForestClassifier())
	return accuracy_score(pipe.predict(x_test),y_test), precision_score(pipe.predict(x_test), y_test), recall_score(pipe.predict(x_test), y_test)
	#To-Do: Implement this function 표준 스케일러와 랜덤포레스트로 이루어진 파이프라인을 만들고 각각을 정확성,정밀도,리콜 순으로 반환했다.

def svm_train_test(x_train, x_test, y_train, y_test):
	svm_pipe = make_pipeline(StandardScaler(),SVC())
	svm_pipe.fit(x_train, y_train)
	return accuracy_score(y_test, svm_pipe.predict(x_test)), precision_score(y_test, svm_pipe.predict(x_test)), recall_score(y_test, svm_pipe.predict(x_test))

	#To-Do: Implement this function sym_pipe에 표준 스케일러와 svm으로 이루어진 파이프라인을 만들고 각각을 정확성,정밀도,리콜 순으로 반환했다


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)