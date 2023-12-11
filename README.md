<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## CS 175: Group 5 Sample Dropout and Variance Reduction in Multi-Agent Environments



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

These are the steps needed to install and run the training. This may vary from system to system and may need extra installations, terminals, or virtual environments to achieve the best results.

### Soccer-Twos Installation

On a Python 3.6+ environment, run:

`pip install soccer-twos`

### MPE Installation

 Using CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

The provided requirement.txt may have redundancy so install each version as necessary for example:

'''
pip install python==1.10.0
'''

``` Bash
# install this package first
pip install seaborn
```

### Soccer-Twos Training

Run the run_soccer_twos_main.py found under soccer-twos-working\mappo-competitive-reinforcement\run_soccer_twos_main.py

Modify the sd_delta, use_sd, use_PSRO elements as needed

### MPE Train
Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results.
Change the .sh file as needed to run the right training.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact
Edward Chang
Nathan Monette
Einar Gatchalian

Project Link: [https://github.com/eddie100971/soccer-twos-working](https://github.com/eddie100971/soccer-twos-working)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [MPE MARL Benchmark On-Policy](https://github.com/marlbenchmark/on-policy )
* [Soccer-Twos Competitive Reinforcement](https://github.com/terran6/mappo-competitive-reinforcement)
* [Soccer-Twos Guide](https://github.com/bryanoliveira/soccer-twos-env)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

