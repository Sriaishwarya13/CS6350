import csv
import math

TRAIN_FILE='./bank/train.csv'
TEST_FILE='./bank/test.csv'


ATTRIBUTES,MST=None,None

def init_attr():
    global columns,ATTRIBUTES,MST
    dataset={}
    anal={}
    reader=csv.reader(open(TRAIN_FILE,'r'))
    for c in columns:
        if c==columns[-1]:
            continue
        dataset[c]=[]
        anal[c]={}
    for sample in reader:
        for i in range(len(columns)-1):
            if sample[i] not in dataset[columns[i]]:
                dataset[columns[i]].append(sample[i])
                anal[columns[i]][sample[i]]=0
            anal[columns[i]][sample[i]]+=1
    mst={}
    for k in anal.keys():
        st=sorted(anal[k].items(), key=lambda x: x[1], reverse=True)
        mm=st[0][0]
        if mm=='unknown':
            mm=st[1][0]
        mst[k]=mm
    ATTRIBUTES,MST = dataset,mst


def def_data(labels):
    dic={}
    for label in labels:
        if label not in dic.keys():
            dic[label]={}
    return dic

def init_data(reader, filt=0):
    global columns,MST
    dataset=[]
    for sample in reader:
        atrs={}
        for i in range(len(columns)):
            if columns[i] not in atrs.keys():
                mm=sample[i]
                if filt==1:
                    if mm=='unknown':
                        mm=MST[columns[i]]
                atrs[columns[i]]=mm
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
        
def Entropy(dic,sum):
    res=0
    for key in dic.keys():
        if key=='sum':
            continue
        res+=Log(dic[key]/sum)
    return res

def ME(dic,sum):
    return sorted(dic.items(), key=lambda x: x[1])[0][1]/sum

def GI(dic,sum):
    res=1
    for key in dic.keys():
        if key=='sum':
            continue
        res-=(dic[key]/sum)**2
    return res

def Gain(entropy, attribute, data, algo='Entropy'):
    gain = entropy
    for value in data[attribute].keys():
        if value == 'sum':
            continue
        
        # Compute the measure based on the specified algorithm
        if algo == 'Entropy':
            nbr = Entropy(data[attribute][value], data[attribute][value]['sum'])
        elif algo == 'ME':
            nbr = ME(data[attribute][value], data[attribute][value]['sum'])
        elif algo == 'GI':
            nbr = GI(data[attribute][value], data[attribute][value]['sum'])
        
        # Adjust the gain by subtracting the weighted measure
        gain -= (data[attribute][value]['sum'] / data['sum']) * nbr
    
    return gain


class Node:
    def __init__(self,label) -> None:
        self.data=label
        self.children={}

def analyse(S):
    global columns
    column_names = []

    # Collect the names of all columns from the first sample
    for label in S[0].keys():
        column_names.append(label)

    attributes = def_data(column_names)
    attributes['sum'] = 0

    # Process each sample to gather attribute data
    for sample in S:
        attributes['sum'] += 1

        for label in column_names:
            if label == columns[-1]:  # Check if it's the target label
                if sample[label] not in attributes[label]:
                    attributes[label][sample[label]] = 0
                attributes[label][sample[label]] += 1
            else:  # Process other attributes
                if sample[label] not in attributes[label]:
                    attributes[label][sample[label]] = {'sum': 0}

                attributes[label][sample[label]]['sum'] += 1

                if sample[columns[-1]] not in attributes[label][sample[label]]:
                    attributes[label][sample[label]][sample[columns[-1]]] = 0
                attributes[label][sample[label]][sample[columns[-1]]] += 1

    return attributes


    
def get_best_attr(attr, algo='Entropy'):
    attribute_list = [label for label in attr.keys() if label != 'sum']
    gain_values = {}

    for attribute in attribute_list:
        if attribute == 'label':
            continue
        if algo == 'Entropy':
            gain_values[attribute] = Gain(Entropy(attr['label'], attr['sum']), attribute, attr, algo='Entropy')
        elif algo == 'ME':
            gain_values[attribute] = Gain(ME(attr['label'], attr['sum']), attribute, attr, algo='ME')
        elif algo == 'GI':
            gain_values[attribute] = Gain(GI(attr['label'], attr['sum']), attribute, attr, algo='GI')

    return sorted(gain_values.items(), key=lambda x: x[1], reverse=True)[0][0]

MAX_LAY = 0


def ID3_algorithm(S, Attributes, level, max_level=0, algo='Entropy'):
    global MAX_LAY
    if level > MAX_LAY:
        MAX_LAY = level

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


def get_val(dic, key):
    if key in dic:
        return dic[key]
    else:
        min_difference = float('inf')
        nearest_key = ''
        for k in dic:
            difference = abs(int(k) - int(key))
            if difference < min_difference:
                min_difference = difference
                nearest_key = k
        return dic[nearest_key]



def predict(root, sample):
    if not root.children:
        return root.data
    else:
        return predict(get_val(root.children, sample[root.data]), sample)


def test(root, S):
    correct_predictions = 0
    total_samples = 0
    for sample in S:
        total_samples += 1
        predicted_value = predict(root, sample)
        if predicted_value == sample[columns[-1]]:
            correct_predictions += 1
    print('Error rate: ' + str((total_samples - correct_predictions) / total_samples))


def run_test(train_data, test_data):
    for i in range(1, 17):
        print(f'Level: {i}')
        print('Entropy: ', end='')
        test(ID3_algorithm(train_data, attributes, 1, i, 'Entropy'), test_data)
        print('Majority Error: ', end='')
        test(ID3_algorithm(train_data, attributes, 1, i, 'ME'), test_data)
        print('Gini Index: ', end='')
        test(ID3_algorithm(train_data, attributes, 1, i, 'GI'), test_data)


def main():
    init_attr()
    train_data_reader=csv.reader(open(TRAIN_FILE,'r'))
    test_data_reader=csv.reader(open(TEST_FILE,'r'))

   
    train_data=init_data(train_data_reader,1)
    test_data=init_data(test_data_reader,1)

    print('run test on testing data:')
    run_test(train_data,test_data)
    print('run test on training data:')
    run_test(train_data,train_data)

if __name__ == '__main__':
    main()
