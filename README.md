# Urban Sound Classification : striving towards a fair comparison

This repo contains code for our paper: [**Urban Sound Classification : striving towards a fair comparison**](https://arxiv.org/pdf/2010.11805.pdf). 

It  provides  a  fair comparison  by  using  the  same  input  representation,  metrics and  optimizer  to  assess  performances.  We  preserve  data  augmentation used by the original papers. We hope this framework could  help  evaluate  new  architectures  in  this  field.


## Environement setup

Python version recquired : 3.6 (higher work). We recommand first to create a new environment in conda/virtualenv then to activate it.

Pip install

~~~bash
pip install -r requirements.txt
~~~

Conda install

~~~bash
conda env create -f environment.yml
~~~

Manual install

~~~bash
pip install numpy scikit-learn pandas tqdm albumentations librosa tensorboard torch torchvision oyaml pytorch-lightning numba==0.49
pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
~~~

## Editing `config.py`

You should edit PATH in `config.py` to match the directory in which everything will be stored.

## Data download and preprocessing

Use the following to download the dataset and precompute inputs.

WARNING : It requires about 30Go of free space.

~~~bash
python data_prep.py --download --mel
~~~

If you want to manualy download and decompress files, you have to put everything in the `audio` directory (`cf config.py`). Then you have to use the above command without the `--download`.

To use relabeling for TALNet modified, copy paste the `best2.csv` into the SONYC-UST folder.

To use transfer learning, download the pretrained models and copy-paste them into the `SONYC-UST/model` folder as shown below.
- the pretrained TALNet on Audioset can be found [here](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/model/TALNet.pt).
- the pretrained CNN10 on Audioset can be found [here](https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1).


Your data folder should look like:

~~~bash
.
└── data                        # Given by PATH in config.py
    ├── SONYC-UST                   
         ├── audio              # Put all audio in it 
         ├── melTALNet          # Contain all computed mel-spectrograms 
         ├── best2.csv 
         │  
         ...
         └── model              # Put TALNet and CNN10 weights here
    ├── ESC-50 
         ├── audio             
         ├── melTALNet 
         ├── meta
         │  
         ...
    ├── UrbanSound8k
         ├── audio             
         ├── melTALNet  
         ├── metadata
         │  
         ...
~~~

## Results and how to reproduce experiments

![Results](img/results.png)

The code should work on both CPU and GPU. If you want to train everything on CPU, remove `gpus=1` in the corresponding model_<model_name>_training.py file. 
The scripts used for the comparison are the test_model_<model_name>.sh file. To run one test, just execute the following command :

~~~bash
sh test_model_<model_name>.sh
~~~

WARNING : CNN10TL and TALNETv3 require the pretrained models on Audioset before running them.

Training results are stored in the data folder:
~~~bash
.
└── data                        # Given by PATH in config.py
    ├── SONYC-UST                   
    ├── ESC-50 
    ├── UrbanSound8k
    └── summaries
         ├── CNN10AllDatasets
         │  
         ...
         └── TFNetAllDatasets
~~~

## Cite

~~~bibtex
@article{ArnaultAnalysis2020,
  title={Urban Sound Classification : striving towards a fair comparison},
  author={Arnault, Augustin and Hanssens Baptiste and Riche, Nicolas},
  journal={arXiv preprint arXiv:2010.11805},
  year={2020}
}
~~~
