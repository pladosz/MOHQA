# MOHQA
This is an implementation of Modulated Hebbian plus Q-learning Architecture (MOHQA) as described in "Reinforcement Learning with Deep Associative Plastic Q-Networks" paper. To run use maze_start.py. The maze environment is included in DynamicMazeEnv.

NOTE: current implementation only works on CUDA enabled machines!

For any queries email: P.ladosz2@lboro.ac.uk



This repository is based on RL repository https://github.com/ShangtongZhang/DeepRL


# Installation instructions:

To install two steps are necessary, first installing MOHQA dependencies then installing the CT-graph environment.

## Installing MOHQA dependencies:

1. (Optional and recomended) Start new Conda environment (for details on installing conda see: [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))
```
conda create -n "ENVNAME" python=3.6
```
2. (Only do this if you did step 1) Activate the environment:
```
conda activate ENVNAME
```
3. Install dependencies using pip:
```
pip install -r requirements.txt
```
## Installing CT-graph:

1. Go into maze ct-graph directory:
```
cd DynamicMazeEnv
```
2. Install the graph:
```
pip install -e .
```

The code should be ready to run.

## Training:
 
 1. Go to the MOHQA directory:
```
cd ..
```
2. Run:
```
python maze_start.py
```


For debugging info use: plot_debugging_info.py. Just define appropriate log directory where indicated on top of that file.


# Acknowledgement

This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. FA8750-18-C-0103.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).

Copyright (C) 2019 Andrea Soltoggio, Pawel Ladosz, Eseoghene Ben-Iwhiwhu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
