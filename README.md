# Urban Sound Classification : striving towards a fair comparison

This repo contains code for our paper: [**Urban Sound Classification : striving towards a fair comparison**](https://arxiv.org/pdf/2010.11805.pdf). It  provides  a  fair comparison  by  using  the  same  input  representation,  metrics and  optimizer  to  assess  performances.  We  preserve  data  augmentation used by the original papers. We hope this framework could  help  evaluate  new  architectures  in  this  field.


## Environement setup

Python version recquired : 3.6 (Higher might work).
We recommand first to create a new environment in conda/virtualenv then to activate it.

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
Your data folder should look like:

~~~bash
.
└── data                        # Given by PATH in config.py
    ├── SONYC-UST                   
         ├── audio              # Put all audio in it 
         ├── melTALNet          # Put all mel-spectrograms in it  
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
    └── summaries
          ├── CNN10AllDatasets
          │  
          ...
          └── TFNetAllDatasets
~~~

## Data download and preprocessing

Use the following to download the dataset and precompute inputs.

WARNING : It requires about 30Go of free space.

~~~bash
python data_prep.py --download --mel
~~~

If you want to manualy download and decompress files, you have to put everything in the `audio` directory (`cf config.py`). Then you have to use the above command without the `--download`.

To use relabeling for TALNet modified, copy paste the `best2.csv` into the SONYC-UST folder.

## Results and how to reproduce experiments

![Results](img/results.png)

The code should work on both CPU and GPU.
If you want to train everything on CPU, remove `gpus=1` in the corresponding model_*.py file. The scripts used for the comparison are the test_model_*.sh file. To run one test, just execute the following command :

~~~bash
sh test_model_MODELNAME.sh
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
