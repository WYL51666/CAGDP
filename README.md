## Dependencies
Install the dependencies via [Anaconda](https://www.anaconda.com/):
+ Python (>=3.9)
+ PyTorch (>=2.0.1)
+ NumPy (>=1.26.1)
+ Scipy (>=1.7.3)
+ tqdm(>=4.66.1)

```python
# create virtual environment
conda create --name CAGDP python=3.9

# activate environment
conda activate CAGDP

#install pytorh from pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

# install other dependencies
pip install -r requirements.txt
```

## Dataset

We provide the weibo dataset in our repository, if you want get other datasets, you can find them in our paper, or you can send email to us, we are pleased to offer you other datasets.

## Usage

Here we provide the implementation of CAGDP along with weibo dataset.

+ To train and evaluate on Weibo:
```python
python run.py -data_name=weibo
```
More running options are described in the codes, e.g., `-data_name= weibo`

## Folder Structure

CAGDP
```
└── data: # The file includes datasets
    ├── weibo
       ├── cascades.txt       # original data
       ├── cascadetrain.txt   # training set
       ├── cascadevalid.txt   # validation set
       ├── cascadetest.txt    # testing data
       ├── idx2u.pickle       # idx to user_id
       ├── u2idx.pickle       # user_id to idx
       
└── models: # The file includes each part of the modules in MetaCas.
    ├── HGATLayer.py # The core source code of Convolution.
    ├── model.py # The core source code of CAGDP.
    ├── TransformerBlock.py # The core source code of time-aware attention.

└── utils: # The file includes each part of basic modules (e.g., metrics, earlystopping).
    ├── EarlyStopping.py  # The core code of the early stopping operation.
    ├── Metrics.py        # The core source code of metrics.
    ├── graphConstruct.py # The core source code of building hypergraph.
    ├── parsers.py        # The core source code of parameter settings. 
└── Constants.py:    
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
└── Optim.py:          # Optimization.

```
## Contact

For any questions please open an issue or drop an email to: `wywu@hdu.edu.cn`
