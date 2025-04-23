# Safe-MPC

This repo contains algorithms for Model Predictive Control, ensuring safety constraints and recursive feasibility.

## NOTICE
Check last version of the code in branch `devel`

## Installation
- Clone the repository\
`git clone https://github.com/idra-lab/safe-mpc.git`
- Install the requirements\
`pip install -r requirements.txt`
- Inside the root folder, download the zip (containing the Neural Network weights) from [here](https://drive.google.com/drive/folders/1RxXyuD6rPAJ7cdMhbY2nh_YfajpJ8Ku-?usp=sharing),
rename it as `nn_models.zip` and unzip it.
- Follow the instructions to install [CasADi](https://web.casadi.org/get/), [Acados](https://docs.acados.org/installation/index.html) and [Pytorch](https://pytorch.org/get-started/locally/).

## Usage 
Run the script `main.py` inside the `scripts` folder. One can consult the help for the available options:
```
cd scripts
python3 main.py --help
```
For example:
- find the initial static configurations of the manipulator on which we run the tests
```
python3 main.py -i
```
- obtain the initial guess (to warm-start the MPC), for a given controller
```
python3 main.py -g -c=receding
```
- run the MPC 
```
python3 main.py --rti -c=receding
```

## References
```bibtex
@misc{lunardi2023recedingconstraint,
      title={Receding-Constraint Model Predictive Control using a Learned Approximate Control-Invariant Set}, 
      author={Gianni Lunardi and Asia La Rocca and Matteo Saveriano and Andrea Del Prete},
      year={2023},
      eprint={2309.11124},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

