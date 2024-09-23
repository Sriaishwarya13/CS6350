import csv
import math

TRAIN_FILE='./car/train.csv'
TEST_FILE='./car/test.csv'

def def_data(labels):
    dic={}
    for label in labels:
        if label not in dic.keys():
            dic[label]={}
    return dic

def init_data(reader):
    global columns
    dataset=[]
    for sample in reader:
        atrs={}
        for i in range(len(columns)):
            if columns[i] not in atrs.keys():
                atrs[columns[i]]=sample[i]
        dataset.append(atrs)
    return dataset

def get_subset(S,attr,value):
    dataset=[]
    for sample in S:
        if sample[attr]==value:
            res={}
            for label in sample.keys():
                if label==attr:
                    continue
                res[label]=sample[label]
            dataset.append(res)
    return dataset

def Log(a):
    return -a*math.log2(a)

def Entropy(dic, total):
    result = 0
    for key in dic:
        if key == 'sum':
            continue
        result += Log(dic[key] / total)
    return result

def ME(dic, total):
    return sorted(dic.items(), key=lambda x: x[1])[0][1] / total

def GI(dic, total):
    result = 1
    for key in dic:
        if key == 'sum':
            continue
        result -= (dic[key] / total) ** 2
    return result

def Gain(entropy, attribute, data, algo='Entropy'):
    gain = entropy
    for value in data[attribute]:
        if value == 'sum':
            continue
        if algo == 'Entropy':
            nbr = Entropy(data[attribute][value], data[attribute][value]['sum'])
        elif algo == 'ME':
            nbr = ME(data[attribute][value], data[attribute][value]['sum'])
        elif algo == 'GI':
            nbr = GI(data[attribute][value], data[attribute][value]['sum'])
        gain -= (data[attribute][value]['sum'] / data['sum']) * nbr
    return gain

class Node:
    def __init__(self,label) -> None:
        self.data=label
        self.children={}

def analyse(S):
    global columns
    column_names = list(S[0].keys())
    attributes = def_data(column_names)
    attributes['sum'] = 0

    for sample in S:
        attributes['sum'] += 1
        for label in column_names:
            if label == columns[-1]:
                attributes[label][sample[label]] = attributes[label].get(sample[label], 0) + 1
            else:
                if sample[label] not in attributes[label]:
                    attributes[label][sample[label]] = {'sum': 0}
                attributes[label][sample[label]]['sum'] += 1
                attributes[label][sample[label]][sample[columns[-1]]] = attributes[label][sample[label]].get(sample[columns[-1]], 0) + 1

    return attributes

def get_best_attr(attr, algo='Entropy'):
    columns = [label for label in attr if label != 'sum']
    gains = {}

    for column in columns:
        if column == 'label':
            continue
        if algo == 'Entropy':
            gains[column] = Gain(Entropy(attr['label'], attr['sum']), column, attr, algo='Entropy')
        elif algo == 'ME':
            gains[column] = Gain(ME(attr['label'], attr['sum']), column, attr, algo='ME')
        elif algo == 'GI':
            gains[column] = Gain(GI(attr['label'], attr['sum']), column, attr, algo='GI')

    return max(gains, key=gains.get)


def ID3_algorithm(S, Attributes, level, max_level=0, algo='Entropy'):
    label_counts = {}

    for sample in S:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    if len(label_counts) == 1:
        return Node(sorted_labels[0][0])
    if not Attributes:
        return Node(sorted_labels[0][0])
    if max_level > 0 and level > max_level:
        return Node(sorted_labels[0][0])

    best_attribute = get_best_attr(analyse(S), algo)
    node = Node(best_attribute)

    for value in ATTRIBUTES[best_attribute]:
        subset = get_subset(S, best_attribute, value)
        if not subset:
            node.children[value] = Node(sorted_labels[0][0])
        else:
            remaining_attributes = [a for a in Attributes if a != best_attribute]
            node.children[value] = ID3_algorithm(subset, remaining_attributes, level + 1, max_level)

    return node


def predict(root,sample):
    if len(root.children)==0:
        return root.data
    else:
        return predict(root.children[sample[root.data]],sample)

def test(root, S):
    correct_predictions = 0
    total_samples = 0
    for sample in S:
        total_samples += 1
        prediction = predict(root, sample)
        #print(prediction + '==' + sample[columns[-1]])
        if prediction == sample[columns[-1]]:
            correct_predictions += 1
    print('Accuracy: ' + str(correct_predictions / total_samples))
    print('Error: ' + str((total_samples - correct_predictions) / total_samples))



def run_test(train_data,test_data):
    for i in range(1,7):
        print('level='+str(i))
        print('Entropy:')
        test(ID3_algorithm(train_data,attributes,1,i,'Entropy'),test_data)
        print('ME:')
        test(ID3_algorithm(train_data,attributes,1,i,'ME'),test_data)
        print('GI:')
        test(ID3_algorithm(train_data,attributes,1,i,'GI'),test_data)

def main():
    with open(TRAIN_FILE, 'r') as train_file, open(TEST_FILE, 'r') as test_file:
        train_data_reader = csv.reader(train_file)
        test_data_reader = csv.reader(test_file)

        train_data = init_data(train_data_reader)
        test_data = init_data(test_data_reader)

    print('Testing on the test data:')
    run_test(train_data, test_data)
    print('Testing on the training data:')
    run_test(train_data, train_data)

if __name__ == '__main__':
    main()
