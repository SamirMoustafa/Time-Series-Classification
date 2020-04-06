# Topological Data Analysis for Time Series Classification
TDA was used to extract topological features from [UCR](http://www.timeseriesclassification.com) time series classification datasets. In our project we argue that topological features may help to identify interesting patterns in data in which shape has meaning. This repo contains the pipeline for extracting the topological features and evaluation of multiple classification algorithms on this features. [Giotto-tda](https://github.com/giotto-ai/giotto-tda), a high-performance topological machine learning toolbox in Python built on top of scikit-learn library, was used to extract topological features from the input data using persistent homology and combine these features with machine learning methods.

<img src="images/TDA.jpg" width="425"/> <img src="images/tda_logo.svg" width="425"/> 	

<img src = "images/Homology.gif" width = "425"/>	

## Repository structure
* [TDA.ipynb](./TDA.ipynb) - the main notebook that demonstrates the application, evaluation and analysis of topological features for time series classification
* [src/TFE](./src/TFE) - contains routines for extracting Persistence Diagram and implemented topological features
* [src/nn](./src/nn) and [src/ae](./src/ae) - contain neural network and VAE implementation
* [src/utils.py](./src/utils.py) - contains helping methods
* [extract_tda_dataset.py](./extract_tda_dataset.py) - script that can be used to generate datasets with topological features from initial UCR datasets
* [evaluation.py](./evaluation.py) - script that can be used for evaluation of extracted topological features datasets
* [Texas_sharpshooter.ipynb](./Texas_sharpshooter.ipynb) - notebook that was used to build Texas Sharpshooter plot
* [CD_diagram.ipynb](./CD_diagram.ipynb) - notebook that was used to build CD diagram

Extracted TDA features datasets from initial UCR datasets can be found [on Google Drive](https://drive.google.com/drive/folders/1GNzazPr4ethNuBNLzQWFy8plZETn5Ckq).

## Setup	
* Clone this repo: 	
```	
git clone https://github.com/SamirMoustafa/Time-Series-Classification.git	
```	
* Install dependencies:	
```	
pip install -r requirements.txt	
```	








# License	
The project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.	
