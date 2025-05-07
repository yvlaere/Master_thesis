# Detection of 5mC Modification in Nanopore Sequencing Data Using Deep Learning

  

A suite of Jupyter notebooks and Python scripts developed to preprocess Oxford Nanopore sequencing signals, train deep learning models for 5-methylcytosine (5mC) detection, and perform thorough performance analyses. The main purpose was the comparison of the facebookresearch ConvNeXt (https://github.com/facebookresearch/ConvNeXt) (https://doi.org/10.1109/CVPR52688.2022.01167) and Google's Transformer (https://doi.org/10.48550/arXiv.1706.03762) architectures. This codebase accompanies my master's thesis. 
## Table of Contents

- Thesis Reference
- Repository Layout
- Getting Started
- Notebook Descriptions
	1. final_E_coli.ipynb
	2. final_HX1.ipynb
	3. calibration_Ecoli.ipynb
	4. calibration_HX1.ipynb
	5. annotation_analyses.ipynb
- References
- License
## Thesis Reference

> **Yari Van Laere** (2022). _Detection of 5mC Modification in Nanopore Sequencing Data Using Deep Learning_. Master's thesis, Ghent University. [Download PDF](https://libstore.ugent.be/fulltxt/RUG01/003/062/276/RUG01-003062276_2022_0001_AC.pdf)

## Repository Layout

```
Master_thesis/
├── data_preprocessing/
│   ├── NSDataset6.py               # Preprocessing ONT data
│   └── WGBSDataset.py              # Preprocessing WGBS data
├── models/
|	├── Convolutional_NN/
|	│   ├── ConvNextBlock1D.py            # ConvNext block for the CNN
|	│   ├── ConvNeXt_model.py             # Old CNN architecture
|	│   ├── ConvNeXt_model_110.py         # Final CNN architecture
|	│   └── TCN_model_110.py              # Old CNN architecture
|	├── Transformer_NN/
|	│   ├── MultiHeadAttention.py         # Multi head attention
|	│   ├── Performer7.py                 # Old transformer architecture 
|	│   ├── TransformerBlock1.py          # Transformer block
|	│   ├── TransformerPreparation.py     # Mask and positional encoding
|	│   ├── Transformer_model62_.py       # Old transformer architecture 
|	│   └── Transformer_model_window16.py # Final transformer architecture
├── notebooks/
│   ├── annotation_analyses.ipynb   # Clustering & annotation analysis
│   ├── calibration_Ecoli.ipynb     # Calibration of E. coli model outputs
│   ├── calibration_HX1.ipynb       # Calibration of HX1 model outputs
│   ├── final_E_coli.ipynb          # E. coli model training and inference
│   └── final_HX1.ipynb             # Human (HX1) model training and inference
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT license file
├── README.md                       # This file
└── thesis.pptx                     # Presentation of thesis highlights            
```

## Getting Started

1. **Clone repository**
    ```
    git clone https://github.com/yvlaere/Master_thesis.git
    cd Master_thesis
    ```
2. **Create and activate Python environment**
    ```
    conda create -n methylation-cnn python=3.8 -y
    conda activate methylation-cnn
    ```
3. **Install required packages**
    ```
    pip install numpy pandas scikit-learn matplotlib torch torchvision jupyter h5py math os random
    ```
## Notebook Descriptions

The `data_preprocessing` and `models` folders contain the necessary data preprocessing and model architecture implementations used in the notebooks. Each notebook drives a stage in the workflow, from raw signal extraction to final performance analysis.

### 1. final_E_coli.ipynb

- The ground truth datasets are datasets where either every cytosine was methylated or datasets where every cytosine was unmethylated
- Prepare the oxford nanopore technologies (ONT) sequencing data to give to the model 
- Train the model (ConvNeXt or Transformer)
- Evaluate the model (ROC curve, PR curve)
- Get predictions
### 2. final_HX1.ipynb

- Prepare whole-genome bisulphite sequencing data (WGBS) as the ground truth
- Prepare the oxford nanopore technologies (ONT) sequencing data to give to the model 
- Train the model (ConvNeXt or Transformer)
- Evaluate the model (ROC curve, PR curve)
- Get predictions
    
### 3. calibration_Ecoli.ipynb`

- Calibrate the predictions of the models using:
	- Isotonic regression
	- Platt scaling
### 4. calibration_HX1.ipynb

- Calibrate the predictions of the models using:
	- Isotonic regression
	- Platt scaling
### 5. annotation_analyses.ipynb

- Compare results of the models for  the human data in the context of genome annotation

## References
- ConvNeXt: Liu, Zhuang & Mao, Hanzi & Wu, Chao-Yuan & Feichtenhofer, Christoph & Darrell, Trevor & Xie, Saining. (2022). A ConvNet for the 2020s. 11966-11976. https://doi.org/10.1109/CVPR52688.2022.01167. 
- Transformer: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). _Attention is all you need_. In _Advances in Neural Information Processing Systems_ (NeurIPS 2017), 30. https://doi.org/10.48550/arXiv.1706.03762

## License

This project is released under the MIT License. See LICENSE for details.
