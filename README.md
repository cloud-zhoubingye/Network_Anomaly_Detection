#  SCNFOD for High-Dimensional Network Anomaly Detection

## Project Overview

This project addresses the challenge of high-dimensional data in network anomaly detection. We propose a hybrid method, **SCNFOD** (Subspace and Deep Feature Network Fusion for Outlier Detection), which combines **SSCOMPOD** (Subspace Learning-based Anomaly Detection) and **DFNO** (Deep Feature Network Optimization). By fusing the anomaly scores from these two methods, SCNFOD achieves superior performance compared to using either method individually.

## Background

High-dimensional data is a common challenge in network anomaly detection, making traditional methods less effective. To tackle this, we leverage two complementary approaches:

1. **SSCOMPOD**: A subspace learning-based method that extracts low-dimensional features from high-dimensional data and computes anomaly scores.
2. **DFNO**: A deep feature optimization method that constructs similarity matrices and optimizes feature representations for anomaly detection.

While each method has its strengths, combining them allows us to exploit their complementary advantages. SCNFOD fuses the anomaly scores from SSCOMPOD and DFNO using a weighted approach.


# Experiment
## CIC-IDS-2017
- [University of New Brunswick. Intrusion Detection Evaluation Dataset (CIC‐IDS2017)[J]. 2017.](https://www.unb.ca/cic/datasets/ids-2017.html)

Results are as follows. Datasets displayed in red font gain better performance than other methods.
<table>
    <tr>
        <th>Sub-dataset</th>
        <th>Params</th>
        <th>AUC of our method</th>
        <th>AUC of DFNO</th>
        <th>AUC of SSC</th>
    </tr>
    <tr>
        <th>Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv</th>
        <th>k=32, Alpha=0.95</th>
        <th>0.5477</th>
        <th>0.1570</th>
        <th>0.5483</th>
    </tr>
    <tr>
        <th>Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv</th>
        <th>k=47, Alpha=0.95</th>
        <th>0.7075</th>
        <th>0.0881</th>
        <th>0.7078</th>
    </tr>
    <tr>
        <th rowspan="3"><span style="color:red; font-weight:bold;">Friday-WorkingHours-Morning.pcap_ISCX.csv</span></th>
        <th>k=2, Alpha=0.05</th>
        <th>0.3680</th>
        <th>0.2954</th>
        <th>0.3560</th>
    </tr>
    <tr>
        <th>k=7, Alpha=0.05</th>
        <th>0.4008</th>
        <th>0.3672</th>
        <th>0.3560</th>
    </tr>
    <tr>
        <th>k=12, Alpha=0.05</th>
        <th>0.4012</th>
        <th>0.3810</th>
        <th>0.3560</th>
    </tr>
    <tr>
        <th>Monday-WorkingHours.pcap_ISCX.csv</th>
        <th>-</th>
        <th>nan</th>
        <th>nan</th>
        <th>nan</th>
    </tr>
    <tr>
        <th>Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv</th>
        <th>-</th>
        <th>nan</th>
        <th>nan</th>
        <th>nan</th>
    </tr>
    <tr>
        <th>Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv</th>
        <th>k=37, Alpha=0.05</th>
        <th>0.1498</th>
        <th>0.2206</th>
        <th>0.0655</th>
    </tr>
    <tr>
        <th rowspan="5"><span style="color:red; font-weight:bold;">Tuesday-WorkingHours.pcap_ISCX.csv</span></th>
        <th>k=22, Alpha=0.05</th>
        <th>0.2281</th>
        <th>0.2192</th>
        <th>0.2188</th>
    </tr>
    <tr>
        <th>k=27, Alpha=0.05</th>
        <th>0.2285</th>
        <th>0.2139</th>
        <th>0.2188</th>
    </tr>
    <tr>
        <th>k=32, Alpha=0.05</th>
        <th>0.2287</th>
        <th>0.2115</th>
        <th>0.2188</th>
    </tr>
    <tr>
        <th>k=37, Alpha=0.05</th>
        <th>0.2271</th>
        <th>0.2129</th>
        <th>0.2188</th>
    </tr>
    <tr>
        <th>k=42, Alpha=0.05</th>
        <th>0.2257</th>
        <th>0.2147</th>
        <th>0.2188</th>
    </tr>
    <tr>
        <th rowspan="10"><span style="color:red; font-weight:bold;">Wednesday-workingHours.pcap_ISCX.csv</span></th>
        <th>k=17, Alpha=0.35</th>
        <th>0.5307</th>
        <th>0.4620</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=22, Alpha=0.35</th>
        <th>0.5316</th>
        <th>0.4653</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=27, Alpha=0.35</th>
        <th>0.5320</th>
        <th>0.4714</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=32, Alpha=0.35</th>
        <th>0.5323</th>
        <th>0.4843</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=37, Alpha=0.15</th>
        <th>0.5345</th>
        <th>0.4860</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=42, Alpha=0.25</th>
        <th>0.5357</th>
        <th>0.4901</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=47, Alpha=0.25</th>
        <th>0.5359</th>
        <th>0.4971</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=52, Alpha=0.25</th>
        <th>0.5356</th>
        <th>0.4999</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=57, Alpha=0.25</th>
        <th>0.5356</th>
        <th>0.5023</th>
        <th>0.5289</th>
    </tr>
    <tr>
        <th>k=62, Alpha=0.25</th>
        <th>0.5353</th>
        <th>0.5023</th>
        <th>0.5289</th>
    </tr>
</table>

## UNSW-NB15
- [Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Results are as follows. Datasets displayed in red font gain better performance than other methods.
<table>
    <tr>
        <th>Sub-dataset</th>
        <th>Params</th>
        <th>AUC of our method</th>
        <th>AUC of DFNO</th>
        <th>AUC of SSC</th>
    </tr>
    <tr>
        <th rowspan="10"><span style="color:red; font-weight:bold;">UNSW-NB15_1.csv</span></th>
        <th>k=2, Alpha=0.25</th>
        <th>0.6696</th>
        <th>0.5552</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=11, Alpha=0.15</th>
        <th>0.6955</th>
        <th>0.6613</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=20, Alpha=0.15</th>
        <th>0.7299</th>
        <th>0.6977</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=29, Alpha=0.15</th>
        <th></th>
        <th>0.7042</th>
        <th>0.6589</th>
    </tr>
</table>

## Unused Datasets
### KDD Cup 1999
- [Cost-based Modeling and Evaluation for Data Mining With Application to Fraud and Intrusion Detection: Results from the JAM Project by Salvatore J. Stolfo, Wei Fan, Wenke Lee, Andreas Prodromidis, and Philip K. Chan](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- Due to unknown reasons, all results shows `nan`.

### CIC IoT Dataset 2023
- [Neto E C P, Dadkhah S, Ferreira R, et al. CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment[J]. Sensors, 2023, 23(13): 5941.](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- This dataset contains almost only attack records, and there is nearly no normal records, so it is not suitable.


### Mapple
- [Q. Li, B. Wang, X. Wen, Y. Chen, Cybersecurity situational awareness framework based on ResNet modeling](https://maple.nefu.edu.cn/)
- The authors of Mapple did not label the dataset, so it can not be used.

# How to Install
- Place datasets in the `dataset` folder.
```
conda create --name myenv python=3.12
conda activate myenv
pip install -r requirements.txt
python main.py
```
- Run the following command to prepare environments.
```
conda create --name myenv python=3.12
conda activate myenv
pip install -r requirements.txt
```
- Run the following command to start the experiments.
```
conda activate myenv
python main.py
```
