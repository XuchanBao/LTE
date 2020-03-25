# LTE
Learning to Elect - Using Neural Networks to Represent Voting Rules

## Installation
* Create a new conda environment and activate it:
    ```
    conda create -n lte python=3.7
    conda activate lte
    ```
    
* Install PyTorch, following instructions in `https://pytorch.org`. 

* Navigate to the root of the project. Install the package, along with requirements:
    ```
    python setup.py install
    ```
  
* Install spaghettini. 
```
pip install git+https://github.com/cemanil/spaghetti.git@master
```

* Instapp nbdev, to make sure ipython notebooks are commitable. 
``` 
pip install nbdev
nbdev_install_git_hooks
```

* Add project root to PYTHONPATH. One way to do this: 
    ```
    export PYTHONPATH="${PYTHONPATH}:`pwd`"
    ``` 
  
* Set up aliases.
``` 
goto_audiocaps="cd <path-to-project>"
```
  