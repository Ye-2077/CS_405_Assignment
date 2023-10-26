import numpy as np

def distanceFunc(metric_type, x_sample, x_train):

    sample = np.array(x_sample)
    X_train = np.array(x_train)

    sample = np.repeat(sample, X_train.shape[0], axis=0)
    train = np.tile(X_train, (x_sample.shape[0], 1))

    delta = sample-train
    if metric_type == "L1":
        L1 = np.sum(np.abs(delta), axis=1, keepdims=True)
        L1 = L1.reshape(x_sample.shape[0], X_train.shape[0])
        distance = L1

    if metric_type == "L2":
        L2 = np.sqrt(np.sum(delta**2, axis=1, keepdims=True))
        L2 = L2.reshape(x_sample.shape[0], X_train.shape[0])
        distance = L2

    if metric_type == "L-inf":
        L3 = np.max(np.abs(delta), axis=1, keepdims=True)
        L3 = L3.reshape(x_sample.shape[0], X_train.shape[0])
        distance = L3

    # construction of distance looks like
    # [L11, L12, L13, ..., L1n] --> sample_1, L with 400 X_trains
    # [L21, L22, L23, ..., L2n] --> sample_2, L with 400 X_trains
    # [...]
    # [Lm1, Lm2, Lm3, ..., Lmn] --> sample_m, L with 400 X_trains

    return distance


def computeDistancesNeighbors(K, metric_type, X_train, y_train, sample):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    sample = np.array(sample)

    distance = distanceFunc(metric_type, sample, X_train)

    neighbors_index = np.argpartition(distance, K, axis=1)[:, :K]
    neighbors_index = np.sort(neighbors_index, axis=1)

    neighbors = y_train[neighbors_index]

    # The construction of neighbors looks like:
    # in case that K = 5
    # [1,0,1,1,1] --> nearst 5 points's lable of sample_1
    # [0,0,1,1,1] --> nearst 5 points's lable of sample_2

    return neighbors


def Majority(neighbors):

    count_0 = np.sum(neighbors == 0, axis=1)
    count_1 = np.sum(neighbors == 1, axis=1)
    predicted_value = np.zeros([neighbors.shape[0], 1])
    # predicted_value = np.zeros(neighbors.shape[0])

    # Perform majority voting
    for idex in range(0, neighbors.shape[0]):
        if count_0[idex] > count_1[idex]:
            predicted_value[idex] = 0
        else:
            predicted_value[idex] = 1

    return predicted_value


def KNN(K, metric_type, X_train, y_train, X_val):

    neighbors = computeDistancesNeighbors(K, metric_type, X_train, y_train, X_val)
    predictions = Majority(neighbors)

    return predictions



def load_data(path):

    # traning samples
    with open(path, "r") as file:
        num_training_samples = int(file.readline().strip())
        training_samples = []
        for _ in range(num_training_samples):
            line = file.readline().strip().split()
            label = int(line[0])
            features = [float(x) for x in line[1:]]
            training_samples.append([label] + features)

    training_samples = np.array(training_samples)

    # testing samples
    with open(path, "r") as file:
        for _ in range(num_training_samples + 1):
            file.readline()

        num_testing_samples = int(file.readline().strip())
        testing_samples = []
        for _ in range(num_testing_samples):
            line = file.readline().strip().split()
            label = int(line[0])
            features = [float(x) for x in line[1:]]
            testing_samples.append([label] + features)

    testing_samples = np.array(testing_samples)

    return training_samples, testing_samples


def load_input():

    N1 = int(input())
    training_samples = []

    for _ in range(N1):
        line = input().split()
        features = [float(x) for x in line[:]]
        training_samples.append(features)

    N2 = int(input())
    testing_samples = []

    for _ in range(N2):
        line = input().split()
        features = [float(x) for x in line[:]]
        testing_samples.append(features)

    training_samples = np.array(training_samples)
    testing_samples = np.array(testing_samples)

    # print(training_samples)
    # print(testing_samples)

    return training_samples, testing_samples


def main():

    K = [3, 5, 7]
    norm = ["L1", "L2", "L-inf"]
    # path = "homework2/testcase_0/0.in"
    # training_samples, testing_samples = load_data(path)
    training_samples, testing_samples = load_input()

    X_train = training_samples[:, 1:]
    y_train = training_samples[:, 0]

    new_order = np.argsort(testing_samples[:, 0])
    new_testing_samples = testing_samples[new_order]
    unique_values, indices = np.unique(new_testing_samples[:, 0], return_inverse=True)
    matrices = np.split(new_testing_samples, np.where(indices[1:] != indices[:-1])[0] + 1)

    prediction = np.array([]).reshape(0,1)
    for i, matrix in enumerate(matrices):
        X_val = matrix[:, 1:]
        pre = KNN(int(matrix[0, 0]), "L2", X_train, y_train, X_val)
        prediction = np.concatenate((prediction, pre))

    # re-order
    restored_order = np.argsort(new_order)
    prediction = prediction[restored_order]
    
    for pre in prediction:
        print(int(pre[0]))


if __name__ == "__main__":
    main()
