# LTE
Learning to Elect - Using Neural Networks to Represent Voting Rules

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
EXP_DIR="runs/030_mimic_plurality_nested_gnn"
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
PROJECT_ROOT="<path_to_project_root>"
EXP_DIR="./runs/003_fixing_plurality_training"
cd ${PROJECT_ROOT}
bash ./src/mains/generate_experiments.sh ${EXP_DIR}
cd ./runs/${EXP_DIR}
sbatch batch_run.sh

```

* IMPORTANT: Make sure to check out src/experiments/generate_sbatch_script.py to use the appropriate global variables. 


## Experiments
### Approximating Plurality. 
``` 
EXP_DIR="runs/005_approx_oracle"
python -m src.mains.train --cfg ${EXP_DIR}/template.yaml
```