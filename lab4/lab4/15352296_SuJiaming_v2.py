#author: simon
#date:   2017-11-3

from math import log
import csv
from numpy import *

# calculate the original entropy of the data set
def cal_entropy(data_set):
	num_entry = len(data_set)
	# count the number of entries of each label
	label_cnt = {}
	for feat_vec in data_set:
		cur_label = feat_vec[-1]
		if cur_label not in label_cnt.keys():
			label_cnt[cur_label] =0
		label_cnt[cur_label] += 1
	entropy = 0.0

	#calculate entropy
	for key in label_cnt:
		posb = float(label_cnt[key]) / num_entry
		entropy += abs(posb * log(posb))
	return entropy

def get_data(file):
	data_set = loadtxt(open(file),delimiter=',')
	return data_set

#according chosen feature to spilit(get) the returned data set
def spilit_data_set(data_set,feat,value):
	return_data_set = []
	for vec in data_set:
		if vec[feat] == value:
			#get rid of the chosen feature
			reduced_vec = vec[:feat]
			reduced_vec.extend(vec[feat+1:])
			return_data_set.append(reduced_vec)
	return return_data_set


# calculate the information gain to choose the best feature as the node
# type decide the method ( ID3 || C4.5 || GINI) to choose feature 
#default method = "c4.5"
def choose_feat(data_set,method="C4.5"):
	num_feat = len(data_set[0])-1
	best_feat = -1
	if method.lower() == "gini":
		best_gini = 1.0
		for i in range(num_feat):
			#substract all the value of feat[i]
			feat_list = [vec[i] for vec in data_set]
			uniq_val = set(feat_list)
			for value in uniq_val:
				new_gini = 1.0
				sub_data_set = spilit_data_set(data_set,i,value)
				posb = float(len(sub_data_set))/len(data_set)
				new_gini -= posb**2
			if new_gini < best_gini:
				best_gini = new_gini
				best_feat = i
		return best_feat

	elif method.lower() == "id3":
		base_entropy = cal_entropy(data_set)
		best_info_gain = 0.0
		for i in range(num_feat):
			#substract all the value of feat[i]
			feat_list = [vec[i] for vec in data_set]
			uniq_val = set(feat_list)
			for value in uniq_val:
				sub_data_set = spilit_data_set(data_set,i,value)
				new_entropy = 0.0
				posb = float(len(sub_data_set))/len(data_set)
				new_entropy += cal_entropy(sub_data_set)

			info_gain = base_entropy - new_entropy
			if info_gain > best_info_gain:
				best_info_gain = info_gain
				best_feat = i
		return best_feat

	else:
		best_info_gain_rate = 0;
		base_entropy = cal_entropy(data_set)
		best_gain_ratio = 0.0
		for i in range(num_feat):
			#substract all the value of feat[i]
			entropy_feat = 0.0
			feat_list = [vec[i] for vec in data_set]
			uniq_val = set(feat_list)
			for value in uniq_val:
				sub_data_set = spilit_data_set(data_set,i,value)
				new_entropy = 0.0
				posb = float(len(sub_data_set))/len(data_set)
				#calculate the entropy of feature i
				entropy_feat += abs(posb*log(posb))
				new_entropy += cal_entropy(sub_data_set)
			info_gain = base_entropy - new_entropy
			new_gain_ratio = info_gain / entropy_feat
			if new_gain_ratio > best_gain_ratio:
				best_gain_ratio = new_gain_ratio
				best_feat = i
		return best_feat



# train_set = get_data('train.csv')
# train_set = train_set.tolist()
train_set = [[1,1,1,0],[2,2,2,0],[3,3,3,1]]
print(choose_feat(train_set,'gini'))
print(choose_feat(train_set,'id3'))
print(choose_feat(train_set,'c4.5'))
