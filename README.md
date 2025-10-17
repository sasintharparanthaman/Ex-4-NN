
<H3> Sasinthar P</H3>
<H3> 212223230199</H3>
<H3>EX. NO.4</H3>
<H3>DATE:17.10.2025</H3>
<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification
<H3>Theory</H3>

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

<H3>Algorithm:</H3>

1. Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7. In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

<H3>Program:</H3> 


```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
df=pd.read_csv('heart.csv')
df
X = df.iloc[:, 0:13]  
y = df.iloc[:, 13] 
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print("\nPredictions:")
print(predictions)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

df1=pd.read_csv('heart.csv')
df1
a=df1.iloc[:,0:13]
b=df1.iloc[:,13:14]
a.head()
b.tail()
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.25,random_state=42)
scaler=StandardScaler()
scaler.fit(a_train)
a_train = scaler.transform(a_train)
a_test = scaler.transform(a_test)
m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500, random_state=42)
m1.fit(a_train, b_train.values.ravel())
predicted_values = m1.predict(a_test)
print("\nPredicted Values:")
print(predicted_values)
print("\nConfusion Matrix:")
print(confusion_matrix(b_test, predicted_values))
print("\nClassification Report:")
print(classification_report(b_test, predicted_values))

```



<H3>Output:</H3>

### Splits training and testing data as 80-20

#### Heart Dataset

<img width="1001" height="556" alt="Screenshot 2025-10-04 192723" src="https://github.com/user-attachments/assets/a44733df-a59a-4ff3-ac52-80f35133e764" />

#### X 

<img width="931" height="556" alt="Screenshot 2025-10-04 192740" src="https://github.com/user-attachments/assets/682ee506-ee38-4420-b850-560b5e36e7cf" />

#### Y

<img width="931" height="556" alt="Screenshot 2025-10-04 192740" src="https://github.com/user-attachments/assets/86580931-afe9-4910-b51a-bff65df7fbca" />

#### Predicted Values

<img width="948" height="236" alt="Screenshot 2025-10-04 192803" src="https://github.com/user-attachments/assets/76910344-cdf3-43c1-ab88-d6742d376a24" />

#### Confusion Matrix

<img width="345" height="140" alt="Screenshot 2025-10-04 192811" src="https://github.com/user-attachments/assets/d9b2729c-50e2-4aa0-bda3-b0a7dcd81a89" />

#### Classification Report

<img width="749" height="324" alt="Screenshot 2025-10-04 192820" src="https://github.com/user-attachments/assets/9e8aa8f8-962e-434c-b7a7-c64a4af6122a" />

### Splits training and testing data as 75-25
#### Heart Dataset

<img width="1031" height="550" alt="Screenshot 2025-10-04 193508" src="https://github.com/user-attachments/assets/d91cbfd7-e0e4-48ee-81f1-fe917c0e10d5" />

<img width="917" height="336" alt="Screenshot 2025-10-04 193532" src="https://github.com/user-attachments/assets/1372644d-8279-4330-bb36-0d88b46f1772" />

<img width="513" height="328" alt="Screenshot 2025-10-04 193546" src="https://github.com/user-attachments/assets/9a5e0463-a787-4dce-8405-af3b54c63892" />

#### Predicted Values

<img width="993" height="262" alt="Screenshot 2025-10-04 193601" src="https://github.com/user-attachments/assets/75f0dbfb-118d-4b5d-89f1-aff3b8511704" />

#### Confusion Matrix
<img width="283" height="144" alt="Screenshot 2025-10-04 193614" src="https://github.com/user-attachments/assets/62f2337a-2d4e-43ef-80c8-b1ac6bb0974b" />

#### Classification Report

<img width="763" height="320" alt="Screenshot 2025-10-04 193623" src="https://github.com/user-attachments/assets/cecce9bc-a5cb-4cc2-8011-5b443ee087c8" />


<H3>Result:</H3>

Thus, MLP is implemented for multi-classification using python.

