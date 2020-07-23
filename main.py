import numpy as np # 1.16.4
import pandas as pd # 0.24.2
import matplotlib.pyplot as plt # 2.2.4
#Python 2.7

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )

def read_and_divide_into_train_and_test(csv_file):
    df = pd.read_csv(csv_file)
    df.drop('Code_number',axis = 1,inplace = True)
    df.loc[df['Bare_Nuclei'] == "?", 'Bare_Nuclei'] = np.nan
    df = df.astype(float)
    
    def calculateMeanValue(df):
        totalSum = df['Bare_Nuclei'].sum()
        totalNum = len(df['Bare_Nuclei'])
        return round(totalSum / totalNum)

    df = df.fillna(value = calculateMeanValue(df))
    
    #train
    training = df.sample(frac=0.8,random_state=200)
    
    training_labels = training.iloc[:, -1].values
    training_inputs = training.iloc[:, :-1].values
    
    #test
    testing = df.drop(training.index)
    
    test_labels = testing.iloc[:, -1].values
    test_inputs = testing.iloc[:, :-1].values

    
    training_labels = training_labels.reshape(len(training_labels),1)

    #visualization
    
    training.drop('Class',axis = 1,inplace = True)
    labels = training.corr().columns
    
    fig, ax = plt.subplots()

    

    im = ax.imshow(training.corr().to_numpy())
    ax.set_xticks(np.arange(len(training.corr().columns)))
    ax.set_yticks(np.arange(len(training.corr().index)))

    ax.set_xticklabels(training.corr().columns)
    ax.set_yticklabels(training.corr().index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(training.corr())):
        for j in range(len(training.corr())):
            ax.text(i, j, round(training.corr().to_numpy()[i,j],2),ha="center", va="center", color = "w")

    ax.set_title("Correlations")
    fig.tight_layout()
    
    cbar=plt.colorbar(mappable=im)

    
    plt.show()
    return training_inputs, training_labels, test_inputs, test_labels



def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    #calculate test_predictions
    test_predictions = sigmoid(np.dot(test_inputs,weights))
    #TODO map each prediction into either 0 or 1
    
    for x in test_predictions:
        if x[0] > 0.5:
            x[0] = 1
        else:
            x[0] = 0
            
    count = 0
    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
        count += 1
    
    accuracy = tp / count
    
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    #todo plot loss and accuracy change for each iteration

    # setting a style to use 
    plt.style.use('fivethirtyeight') 
    fig = plt.figure()

    # defining subplots and their positions     
    plt1 = fig.add_subplot(211) 
    plt2 = fig.add_subplot(212) 
    plt1.plot()
    plt1.plot(loss_array,color ='r',label = 'Loss Change')
    plt1.set_title('Loss Change') 
    plt1.legend()

    plt2.plot(accuracy_array,color ='g',label = 'Accuracy Change')
    plt2.set_title('Accuracy Change') 
    plt2.legend()
    fig.subplots_adjust(hspace=.5,wspace=0.5) 
    plt.show()


def main():
    csv_file = 'breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        #calculate outputs
        outputs = np.dot(training_inputs,weights)
        outputs = sigmoid(outputs)
        #calculate loss
        loss = training_labels - outputs
        #calculate tuning
        tuning = loss * sigmoid_derivative(outputs)
        
        #update weights
        weights += np.dot(training_inputs.T, tuning)
        ################weights += np.dot(np.transpose(training_inputs),tuning)
        
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
        loss_array.append(loss.mean())
        
    # you are expected to add each accuracy value into accuracy_array
    # you are expected to find mean value of the loss for plotting purposes and add each value into loss_array
    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()
