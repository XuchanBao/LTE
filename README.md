# LTE
Code for the paper ["Learning to Elect"](https://arxiv.org/abs/2108.02768).
```
@inproceedings{NEURIPS2021_42d6c7d6,
 author = {Anil, Cem and Bao, Xuchan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {8006--8017},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Elect},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/42d6c7d61481d1c21bd1635f59edae05-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
## Installation
* Create a new conda environment and activate it:
```
conda create -n lte python=3.7
conda activate lte
```
    
* Install PyTorch, following instructions in [Pytorch website](https://pytorch.org). 

* Install Deep Graph Library (dgl) following instructions on its website. 

* Install spaghettini. 
```
pip install git+https://github.com/cemanil/spaghetti.git@master
```

* Navigate to the root of the project. Install the package, along with requirements:
```
python setup.py install
```

# Install cvxpy with GLPK bindings (for solving Kemeny elections efficiently)/
``` 
pip install cvxopt
CVXOPT_BUILD_GLPK=1
CVXOPT_GLPK_LIB_DIR=/path/to/glpk-X.X/lib
CVXOPT_GLPK_INC_DIR=/path/to/glpk-X.X/include
pip install cvxopt
```

* Install nbdev, to make sure ipython notebooks are commitable. 
``` 
pip install nbdev
nbdev_install_git_hooks
```

* Add project root to PYTHONPATH. One way to do this: 
```
export PYTHONPATH="${PYTHONPATH}:`pwd`"
``` 
 
## Training
### Step 1: Config
Find a sample config in `runs/001_find_max/template.yaml`. 

### Step 2: Training 
You can run the following commands to train the model. The logger will automatically log the training progress. 
``` 
EXP_DIR="runs/ICML/mimicking/deep_sets"
python -m src.mains.train --cfg ${EXP_DIR}/template.yaml
```

You can remove the results of the experiment using:
``` 
cd ${EXP_DIR}
rm -rf lte wandb
cd -

```

### Step 3: Generating batch experiments. 
* Create a new experiment directory (i.e. `EXP_DIR="./runs/003_fixing_plurality_training"`)
* Copy a previous config in new experiment directory and modify it for a batched experiment. This template has to be
called "template". 
* Run:
```
EXP_DIR="runs/ICML/mimicking_kemeny/deep_sets"
bash ./src/mains/generate_experiments.sh ${EXP_DIR} t4v2
sbatch ${EXP_DIR}/batch_run.sh

```

* Inspecting generated slurm files. 
``` 
ls ${EXP_DIR}
cat ${EXP_DIR}/sl
```

* Clearing results, if needed. 
``` 
echo ${EXP_DIR}
cd ${EXP_DIR}
rm -rf batch_run.sh results
cd -

```

* IMPORTANT: Make sure to check out src/experiments/generate_sbatch_script.py to use the appropriate global variables. 


## Experiments
### Approximating Plurality. 
``` 
EXP_DIR="runs/005_approx_oracle"
python -m src.mains.train --cfg ${EXP_DIR}/template.yaml
```
