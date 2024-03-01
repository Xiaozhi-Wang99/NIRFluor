# NIRFluor: a deep learning platform for predicting optical properties of small molecule near-infrared fluorophores
## About
Small molecule near-infrared (NIR) fluorophores play a crucial role in disease diagnosis and early detection of various markers in living organisms. To accelerate their development and design, a deep learning platform, NIRFluor, using an experimental big database and SOTA deep learning method (MT-FinGCN) to predict six optical properties of a given NIR organic small molecule in different solvents with different ratios, including absorption wavelength, emission wavelength, stokes shift, extinction coefficient, photoluminescence quantum yield, and lifetime.

-------------
![image](https://github.com/Xiaozhi-Wang99/NIRFluor/blob/main/image/Figure1.png)
#### Development flowchart of the NIRFluor platform. 
-------------
![image](https://github.com/Xiaozhi-Wang99/NIRFluor/blob/main/image/Figure2.png)
#### Distribution of datasets and performance of identification models. 

-------------
![image](https://github.com/Xiaozhi-Wang99/NIRFluor/blob/main/image/Figure3.png)
#### Performance of prediction models and scatter plots of predictions. 

-------------
![image](https://github.com/Xiaozhi-Wang99/NIRFluor/blob/main/image/Figure4.png)
#### Explainability of ST-GCN-non-SF model on the absorption wavelength prediction. 

-------------
## Packages:
1. Python (version: 3.8.18)
2. PyTorch (version: 2.0.1)
3. Numpy (version: 1.24.3)
4. Pandas (version: 1.5.3)
5. Pytorch Geometric (version: 2.4.0)
6. Scikit-learn (version: 1.3.0)
7. xgboost (version: 1.7.3)
