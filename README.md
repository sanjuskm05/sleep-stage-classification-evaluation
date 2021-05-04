# sleep-stage-classification-evaluation

Deep learning methods exhibit promising performance for predictive modeling in healthcare. Multiple researchers have applied different deep learning models for sleep stage scoring for patients and improve the accuracy close to that of human experts.
We have implemented a Convolution Neural Network (CNN) based architecture which has 78% prediction accuracy with multi-channel sleep electroencephalogram (EEG) data. This result is very promising and can be extended by applying additional layers to the CNN and adding a Recurrent Neural Network to capture the temporal nature of the sleep data.

# Data Description

Physiological signals acquired during sleep, like electroencephalography (EEG), can be used to computationally determine sleep apnea, cardiovascular disorders, sleep stage annotations, etc. We can create an efficient sleep detector,
provided we can get good accuracy for the sleep stage classification.
For our project, we have used the PhysioNet dataset. This sleep European Data Format (EDF) dataset consists of 197 whole night Polysomnographic sleep recordings, containing EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature. Corresponding hypnograms (sleep patterns) were manually scored by well-trained sleep technicians.
The total size of data is 8.1GB in compressed format. There are two kinds of files for the subjects, one is data and the other is the annotation files.
1. The data files (*PSG.edf files) are whole night PSG data containing EEG (from Fpz-Cz and Pz-Oz electrode locations), EOG (horizontal), submental chin EMG, and an event marker. It might also contain the oro-nasal respiration and rectal body temperature
2. The other set of files called hypnogram files have the sleep data annotated by the experts
The data stored in PSG is in EDF format which is standard for the EEG and PSG signals. EDF+ is an extension of the EDF format and it usually works with the EDF format. The specification of the EDF format can be found at,
https://www.edfplus.info/specs/edfplus.html#PSGwithMSLT

# Data Processing
We must consider the below approaches while preprocessing the data before model development
1. We must use the mne and the pyedflib python library to read the EDF files as it comes with their basic functionalities
2. some of the data is missing i.e., for the first nights of subjects 36 and 52 and the second night for subject 13. We will have to take care of this while processing ad ignore that.
3. Normalization of the data is also needed for mean zero and variance 1 which will help to eliminate the errors which we might have during the training process
4. We will have to see the class imbalance problems as there will be many classes where the no of stages is more
5. The dataset consists of sleep stages W, R, N1, N2, N3, N4, M (Movement time), and ‘?’ (not scored). So, for our work, we will try to remove M and ‘?’ data while processing the data.
6. We have checked if the class labels are balanced in our dataset

![image](https://user-images.githubusercontent.com/8688478/116949589-b5a85080-ac50-11eb-9c32-36951d78467a.png)

# CNN Architecture
A convolutional neural network (CNN) is composed of multiple convolutional(filtering) and pooling (sub-sampling) layers with a form of non-linearity applied before or after pooling. These layers are often followed by one or more fully connected layers. In a multi-class classification application such as the sleep stage scoring the last layer of a CNN is often a softmax layer. The feature selection is done automatically using a CNN and then those features are fed to one/more linear layers for classification. The CNNs are trained using iterative optimization with the backpropagation algorithm. The optimization method used in this paper is stochastic gradient descent (SGD).
We have implemented the preliminary model as in Fig.3 as a CNN based on the architecture suggested in the paper by O. Tsinalis et al. This model consists of 2 pairs of convolution and pooling layers followed by 2 fully connected layers. The signal enters as a 1-dimensional data which goes through 1-D convolution layer. The output then gets stacked and passed through a 2-D convolution layer. After every convolution operation non-linearity is applied through a rectified linear unit (ReLU) and downsampled using max-pooling. Post convolution operation the signals passes through 2 fully connected layers.

![image](https://user-images.githubusercontent.com/8688478/116949679-facc8280-ac50-11eb-80e1-c2d7b7533eeb.png)

The cost function used for training this model is Cross-Entropy loss which implicitly applies the softmax function on the signal. We started with a low learning rate of 0.0001 which is reduced progressively so that the model does not overfit.
We have implemented the CNN architecture using the Python Library PyTorch 1.8.1.

# Metrics
We have computed different sets of metrics during the training as well as during validation. During training, we have computed accuracy and the F1-score. During validation, we computed Cohen-Kappa, AUCROC, accuracy, and F1-score to measure the performance of the model.
We have chosen F1-score which is a harmonic mean of precision and sensitivity. F1-score is a more comprehensive model performance measure than precision and sensitivity themselves as they cannot be improved at the same time.


# EXPERIMENTAL RESULTS

We are trying to improve the accuracy by implementing two changes on the base model, 
1. Combining data from fpz-cz and pz-oz for training & prediction and 
2. Increasing patients’ records to 50 for generating the below results.
We have benchmarked our results by training the model using data from fpz-cz and pz-oz channels. We have also combined these two channels’ data to compute the evaluation metrics.


