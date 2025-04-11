#  SCNFOD for High-Dimensional Network Anomaly Detection

## Project Overview

This project addresses the challenge of high-dimensional data in network anomaly detection. We propose a hybrid method, **SCNFOD** (Subspace and Deep Feature Network Fusion for Outlier Detection), which combines **SSCOMPOD** (Subspace Learning-based Anomaly Detection) and **DFNO** (Deep Feature Network Optimization). By fusing the anomaly scores from these two methods, SCNFOD achieves superior performance compared to using either method individually.

## Background

High-dimensional data is a common challenge in network anomaly detection, making traditional methods less effective. To tackle this, we leverage two complementary approaches:

1. **SSCOMPOD**: A subspace learning-based method that extracts low-dimensional features from high-dimensional data and computes anomaly scores.
2. **DFNO**: A deep feature optimization method that constructs similarity matrices and optimizes feature representations for anomaly detection.

While each method has its strengths, combining them allows us to exploit their complementary advantages. SCNFOD fuses the anomaly scores from SSCOMPOD and DFNO using a weighted approach.

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

# Experiment
Our experiments are conducted on three datasets: **CIC-IDS-2017**, **UNSW-NB15**, and **KDD Cup 1999**. The results are summarized in the following tables, or you can download [here](results/results.png).

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>SubDataset Files</th>
      <th colspan="1">SCNFOD</th>
      <th colspan="2">DFNO</th>
      <th colspan="2">SSC</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>AUC</th>
      <th>AUC</th>
      <th>Gain (%)</th>
      <th>AUC</th>
      <th>Gain (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">CIC-IDS-2017</td>
      <td>Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv</td>
      <td>0.5477</td>
      <td>0.1570</td>
      <td>248.8535</td>
      <td>0.5483</td>
      <td>-0.0011</td>
    </tr>
    <tr>
      <td>Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv</td>
      <td>0.7075</td>
      <td>0.0881</td>
      <td>703.0647</td>
      <td>0.7078</td>
      <td>-0.0004</td>
    </tr>
    <tr>
      <td>Friday-WorkingHours-Morning.pcap_ISCX.csv</td>
      <td>0.4008</td>
      <td>0.3672</td>
      <td>9.1503</td>
      <td>0.3560</td>
      <td>0.1258</td>
    </tr>
    <tr>
      <td>Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv</td>
      <td>0.1498</td>
      <td>0.2206</td>
      <td>-32.0943</td>
      <td>0.0655</td>
      <td>1.2870</td>
    </tr>
    <tr>
      <td>Tuesday-WorkingHours.pcap_ISCX.csv</td>
      <td>0.2287</td>
      <td>0.2115</td>
      <td>8.1324</td>
      <td>0.2188</td>
      <td>0.0452</td>
    </tr>
    <tr>
      <td>Wednesday-workingHours.pcap_ISCX.csv</td>
      <td>0.5316</td>
      <td>0.4653</td>
      <td>14.2489</td>
      <td>0.5289</td>
      <td>0.0051</td>
    </tr>
    <tr>
      <td></td>
      <td><b>Avg Gain (Compared to SCNFOD, %)</b></td>
      <td></td>
      <td></td>
      <td><b>158.5593</b></td>
      <td></td>
      <td><b>0.2436</b></td>
    </tr>
    <tr>
      <td rowspan="4">UNSW-NB15</td>
      <td>UNSW-NB15_1.csv</td>
      <td>0.7900</td>
      <td>0.7561</td>
      <td>4.4835</td>
      <td>0.6589</td>
      <td>0.1990</td>
    </tr>
    <tr>
      <td>UNSW-NB15_2.csv</td>
      <td>0.4515</td>
      <td>0.4262</td>
      <td>5.9362</td>
      <td>0.4511</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <td>UNSW-NB15_3.csv</td>
      <td>0.4240</td>
      <td>0.2887</td>
      <td>46.8653</td>
      <td>0.4246</td>
      <td>-0.0014</td>
    </tr>
    <tr>
      <td>UNSW-NB15_4.csv</td>
      <td>0.4081</td>
      <td>0.2820</td>
      <td>44.7163</td>
      <td>0.4090</td>
      <td>-0.0022</td>
    </tr>
    <tr>
      <td></td>
      <td><b>Avg Gain (Compared to SCNFOD, %)</b></td>
      <td></td>
      <td></td>
      <td><b>25.5003</b></td>
      <td></td>
      <td><b>0.0491</b></td>
    </tr>
    <tr>
      <td>KDD Cup 1999</td>
      <td>kddcup.data.corrected.csv</td>
      <td>0.1364</td>
      <td>0.1041</td>
      <td>31.0279</td>
      <td>0.1351</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <td></td>
      <td><b>Avg Gain (Compared to SCNFOD, %)</b></td>
      <td></td>
      <td></td>
      <td><b>31.0279</b></td>
      <td></td>
      <td><b>0.0096</b></td>
    </tr>
  </tbody>
</table>

# Raw Data of Experiments
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
        <th rowspan="8"><span style="color:red; font-weight:bold;">UNSW-NB15_1.csv</span></th>
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
        <th>0.7433</th>
        <th>0.7042</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=38, Alpha=0.15</th>
        <th>0.7558</th>
        <th>0.7332</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=47, Alpha=0.15</th>
        <th>0.7719</th>
        <th>0.7497</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=56, Alpha=0.15</th>
        <th>0.7835</th>
        <th>0.7538</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th>k=65, Alpha=0.15</th>
        <th>0.7900</th>
        <th>0.7561</th>
        <th>0.6589</th>
    </tr>
    <tr>
        <th rowspan="4"><span style="color:red; font-weight:bold;">UNSW-NB15_2.csv</span></th>
        <th>k=38, Alpha=0.35</th>
        <th>0.4515</th>
        <th>0.4262</th>
        <th>0.4511</th>
    </tr>
    <tr>
        <th>k=47, Alpha=0.25</th>
        <th>0.4528</th>
        <th>0.4399</th>
        <th>0.4511</th>
    </tr>
    <tr>
        <th>k=56, Alpha=0.15</th>
        <th>0.4547</th>
        <th>0.4496</th>
        <th>0.4511</th>
    </tr>
    <tr>
        <th>k=65, Alpha=0.15</th>
        <th>0.4564</th>
        <th>0.4530</th>
        <th>0.4511</th>
    </tr>
    <tr>
        <th>UNSW-NB15_3.csv</span></th>
        <th>k=2, Alpha=0.9</th>
        <th>0.4240</th>
        <th>0.2887</th>
        <th>0.4246</th>
    </tr>
    <tr>
        <th>UNSW-NB15_4.csv</span></th>
        <th>k=2, Alpha=0.9</th>
        <th>0.4081</th>
        <th>0.2820</th>
        <th>0.4090</th>
    </tr>
</table>

## Unused Datasets
### KDD Cup 1999
- [Cost-based Modeling and Evaluation for Data Mining With Application to Fraud and Intrusion Detection: Results from the JAM Project by Salvatore J. Stolfo, Wei Fan, Wenke Lee, Andreas Prodromidis, and Philip K. Chan](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
<table>
    <tr>
        <th>Sub-dataset</th>
        <th>Params</th>
        <th>AUC of our method</th>
        <th>AUC of DFNO</th>
        <th>AUC of SSC</th>
    </tr>
    <tr>
        <th><span style="color:red; font-weight:bold;">kddcup.data.corrected.csv</span></th>
        <th>k=2, Alpha=0.9</th>
        <th>0.1364</th>
        <th>0.1041</th>
        <th>0.1351</th>
    </tr>
</table>

### CIC IoT Dataset 2023
- [Neto E C P, Dadkhah S, Ferreira R, et al. CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment[J]. Sensors, 2023, 23(13): 5941.](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- This dataset contains almost only attack records, and there is nearly no normal records, so it is not suitable.


### Mapple
- [Q. Li, B. Wang, X. Wen, Y. Chen, Cybersecurity situational awareness framework based on ResNet modeling](https://maple.nefu.edu.cn/)
- The authors of Mapple did not label the dataset, so it can not be used.
