# MintNet – Automated sleep stage scoring using multi-channel EEG data

Deep learning methods exhibit promising performance for predictive modeling in healthcare. Multiple researchers have applied different deep learning models for sleep stage scoring for patients and improve the accuracy close to that of human experts.
We have implemented a Convolution Neural Network (CNN) based architecture which has 78% prediction accuracy with multi-channel sleep electroencephalogram (EEG) data. This result is very promising and can be extended by applying additional layers to the CNN.

# Problem statement and motivation

Healthcare is an important subject in USA as USA spends a lot of money on healthcare. With the advancement in hardware and cloud technologies Machine learning has become a reality. It is increasingly becoming popular in healthcare as it can be used to provide better care of patients by providing better treatment diagnosis. Among many topics in healthcare, Sleep is a very important and essential part in human life. Sleep is categorized in different stages as per American medical association (Wake, N1, N2, N3, REM).It is very important to accurately and automatically classify the sleep of an individual in these different stages due to below points:

* To identify the sleep related conditions including and but not limited to fatigue, drowsiness , sleep disorders like apnea, insomnia , narcolepsy ,heart disorders.
* Secondly manual sleep scoring is time consuming and prone to human errors.
* Moreover, smart EEG machines can reduce the dependency on the expert technicians and thus saving their time
* It can also help in creating of gadgets where people can themselves proactively monitor and even the further analysis can be sent directly to doctors

The idea is to create an automatic classifier which can take a patient’s sleep data recording and can classify into (Wake, N1, N2, N3, REM) states. For this problem we are planning to take the PhysioNet data set. It contains 197 whole-night Polysomnographic sleep recordings, containing EEG, EOG, chin EMG, and event markers. Some records also contain respiration and body temperature.
Source : https://www.physionet.org/content/sleep-edfx/1.0.0/

![image](https://user-images.githubusercontent.com/8688478/117451932-29788080-af11-11eb-9361-70f7e42ae720.png)

![image](https://user-images.githubusercontent.com/8688478/117451980-38f7c980-af11-11eb-8b70-10fd68770e43.png)


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
We considered the below approaches while preprocessing the data before model development
1. Creating NPZ files by filtering above mentioned data by reading above channels for 30 seconds interval using EDF reader from deep sleepnet paper.
2. Removing the incomplete data for the subject 13,26 and 52.
3. Removing the Movement and :? where the data is not recorded properly
4. Snapshot of the distribution of classes after data processing. 
5. Data is split into 80:20 for training and validation with optimal batch size of 50 based on experimentation.

![image](https://user-images.githubusercontent.com/8688478/117451646-c25acc00-af10-11eb-852e-ce51b3dd5f5d.png)


# CNN Architecture
A convolutional neural network (CNN) is composed of multiple convolutional(filtering) and pooling (sub-sampling) layers with a form of non-linearity applied before or after pooling. These layers are often followed by one or more fully connected layers. In a multi-class classification application such as the sleep stage scoring the last layer of a CNN is often a softmax layer. The feature selection is done automatically using a CNN and then those features are fed to one/more linear layers for classification. The CNNs are trained using iterative optimization with the backpropagation algorithm. The optimization method used in this paper is stochastic gradient descent (SGD).
We have implemented the preliminary model as in below figure as a CNN based on the architecture suggested in the paper by O. Tsinalis et al. This model consists of 2 pairs of convolution and pooling layers followed by 2 fully connected layers. The signal enters as a 1-dimensional data which goes through 1-D convolution layer. The output then gets stacked and passed through a 2-D convolution layer. After every convolution operation non-linearity is applied through a rectified linear unit (ReLU) and downsampled using max-pooling. Post convolution operation the signals passes through 2 fully connected layers.

![image](https://user-images.githubusercontent.com/8688478/117450725-9c80f780-af0f-11eb-95a6-286ce99b5a38.png)


The cost function used for training this model is Cross-Entropy loss which implicitly applies the softmax function on the signal. We started with a low learning rate of 0.0001 which is reduced progressively so that the model does not overfit.
We have implemented the CNN architecture using the Python Library PyTorch 1.8.1.

# MintNet

![image](https://user-images.githubusercontent.com/8688478/117475423-03aba580-af2a-11eb-96d6-e495f5c7432c.png)


# Metrics
We have computed different sets of metrics during the training as well as during validation. During training, we have computed accuracy and the F1-score. During validation, we computed Cohen-Kappa, AUCROC, accuracy, and F1-score to measure the performance of the model.
We have chosen F1-score which is a harmonic mean of precision and sensitivity. F1-score is a more comprehensive model performance measure than precision and sensitivity themselves as they cannot be improved at the same time.


# EXPERIMENTAL RESULTS

We are trying to improve the accuracy by implementing two changes on the base model, 
1. Combining data from fpz-cz and pz-oz for training & prediction and 
2. Increasing patients’ records to 50 for generating the below results.
We have benchmarked our results by training the model using data from fpz-cz and pz-oz channels. We have also combined these two channels’ data to compute the evaluation metrics.


# Licence
* For academic and non-commercial use only
* Apache License 2.0
