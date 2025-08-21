# Safe-MPC

This repo contains algorithms for Model Predictive Control, ensuring safety constraints and recursive feasibility. It is a general framework that can work with any manipulator. However for each system and environment (customizable in config.yaml) a network that approximate a control invariant set is needed.

Until now we tested the framework with the Unitree Z1 robot. URDF files and networks for the sets for some robot-environments settings can be downloaded [here](https://drive.google.com/drive/folders/1qYyiK0fJ9Na2y4qfXjis64UNJ4L7IwVC?usp=sharing). Place the folder robots and nn_models in the root folder and follow the instruction in nn_models to use the networks properly.

## NOTICE
Check last version of the code in branch `devel`

## Installation
- Clone the repository\
`git clone https://github.com/idra-lab/safe-mpc.git`
- Move on the branch devel
- Install the requirements\
`pip install -r requirements.txt`
- Follow the instructions to install [CasADi](https://web.casadi.org/get/), [Acados](https://docs.acados.org/installation/index.html) and [Pytorch](https://pytorch.org/get-started/locally/).
- In root folder, create a folder called robots, and place in it the needed subfolders containing the URDF files of the robots 
- Create also a folder named nn_models, in which are located the networks approximating the control invariant sets.

## Usage 
- In config.yaml, set the simulation parameters
- Run guess_acados.py (-- help to get instructions about the line command arguments) to find the initial configuration and the warm start trajectories. In guess_acados set also the type of cost and task defining cost_controller, and choosing one of the classes defined in cost_definition.
- Run mpc.py to execute the RTI MPC control. Also here set the appropriate cost_controller.

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

