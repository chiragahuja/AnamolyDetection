## Problem Statement
Detecting malicious events in large unlabeled datasets(Anamoly Detection).

## Data
Two comma-separated value files containing extracted [NetFlow](https://www.auvik.com/franklymsp/blog/netflow-basics/) data are provided (`train.csv` and `test.csv`). Both of which have the same schema:

column|type|description
---|---|---
event_id|int|The event id for each row
protocol|str|Network protocol
flow_duration|float|Flow duration
total_fwd_packets|float|Total number of the packets transferred in the forward direction
total_backward_packets|float|Total number of packets transferred in the backward direction
total_length_of_fwd_packets|float|Total length of the packets transferred in the forward direction
total_length_of_bwd_packets|float|Total length of the packets transferred in the backward direction

There is no malicious activity in the training data however there are a small number of events which correspond to an infiltration attack on the network in the test data.

## Solution
- First the dimensions of Dataset is reduced using PCA.
- Then, Applied Unsupervised nearest neighbour algorithm on training data set.
- Then, assuming that 99% of samples in training data are inlier, find the threshold within
  which 99% of data points exists. This threshold value is used to find the outliers in test data.
- Output is then saved in csv file "output.csv".

## Execution steps
- Code is executed using following command:
    python AnamolyDetection.py
- This command will build,train the model using training dataset and then get the outliers in
  test data set.
- An output file "output.csv" is generated in which a label is assigned to each event. Label
  "False" signifies that event is deemed benign and label "True" signifies that event is malicious. The score corresponding to each event represents how malicous an event is. High value represents highly malicious sample. 
