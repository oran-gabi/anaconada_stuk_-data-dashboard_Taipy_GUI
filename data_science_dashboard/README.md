🧠 Overview
A stock data dashboard built using Taipy GUI, with support for machine learning predictions using:

Linear Regression

KNN (K-Nearest Neighbors)

RNN (Recurrent Neural Network) (if TensorFlow is available)

📦 Main Components
📊 Data
Loads S&P 500 stock prices and company information from CSV files.

Filters stock data based on selected country, company, and date range.

🎨 User Interface (GUI)
Created using taipy.gui.builder:

Logo and title

Date range selector

Dropdowns for country and company

Dynamic chart of stock values (plotly)

Prediction display for selected company

🧮 Prediction Models
Each model uses a sliding window of historical data to predict the next value:

get_lin() – Linear Regression

get_knn() – KNN Regressor

get_rnn() – Simple RNN using Keras (if TensorFlow is installed)

📈 Graph Plotting
build_graph_data() – Prepares stock values for selected company(ies)

display_graph() – Uses Plotly to show line charts of stock performance

⚙️ Taipy Configuration
Defines data nodes, tasks, and scenarios:

Each model and data transformation is encapsulated as a task

Connected into a scenario for execution and update on GUI interaction

⚠️ Other Highlights
Handles missing TensorFlow gracefully

Uses icons for UI clarity

Modular function design with clear I/O

# data_science_dashboard

dynamic data science dashboard with Taipy Scenarios

## Introduction

This repository stores the code of a Full Stack GUI App Project featured in my <a href="https://youtu.be/hxYIpH94u20" target="_blank">Youtube tutorial</a>.
<br>
<br>
<img src="https://github.com/user-attachments/assets/0e9cc143-1eb1-432f-a476-f53921b63335" width=600px>

## Requirements
- `taipy >= 4.0.0` (previous versions do no have `tp.Orchestrator()` but use `tp.Core()` instead)
- `plotly == 5.24.1`
- `tensorflow == 2.18.0`
- `scikit-learn == 1.5.2`

### Recommended Environment Setup - No GPU
```
>> conda create -n your_name python=3.11
>> conda activate your_name
>> pip install taipy
>> pip install plotly
>> pip install tensorflow
>> pip install scikit-learn
```

### Recommended Environment Setup - with GPU
This enviropnment will result in obtaining predictions faster. How faster depends on your CPU and GPU model.
<br>
On my end, it results in approx. 20% speed up from switching to RTX 4080 GPU over 12th Gen Intel i9-12900k CPU.
<br>
<br>
*Please replace the first command with one that <b>matches your system requirements and CUDA version</b> from <a href="https://docs.rapids.ai/install/" target="_blank">the official RAPIDS installation guide</a>. 
```
>> conda create -n your_name -c rapidsai -c conda-forge -c nvidia cudf=24.10 python=3.12 'cuda-version>=12.0,<=12.5'
>> conda activate your_name
>> pip install taipy
>> pip install plotly
>> pip install tensorflow
>> pip install scikit-learn
```

## Dataset
S&P 500 Stocks (daily updated) by Larxel
<br>
https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

## Further Learning
Please checkout Taipy's <a href="https://links.taipy.io/Mariya" target="_blank">Official Github Repo</a> for more details and contribution guidelines.

## Connect with me
<b>⭐ YouTube</b>
<br>
     https://youtube.com/@pythonsimplified
<br>
<b>⭐ Discord</b>
<br>
     https://discord.com/invite/wgTTmsWmXA
<br>
<b>⭐ LinkedIn</b>
<br>
     https://ca.linkedin.com/in/mariyasha888
<br>
<b>⭐ Twitter</b>
<br>
     https://twitter.com/mariyasha888
<br>
<b>⭐ Blog</b>
<br>
     https://www.pythonsimplified.org


