# GVGAI for L2M

Setup guide for [GVGAI](http://www.gvgai.net/) ([code](https://github.com/rubenrtorrado/GVGAI_GYM), [Gym](https://github.com/rubenrtorrado/GVGAI_GYM)). More detailed documantation on GVGAI can be found on their [wiki](https://github.com/EssexUniversityMCTS/gvgai/wiki).

## Motivation
GVGAI provides a set of games of varying complexity that is compatible with the OpenAI Gym and allows us to evaluate simple tests of the various modules.  Sebastian Risi's recent [work](https://arxiv.org/abs/1806.10729) ([code](https://github.com/njustesen/a2c_gvgai)) on automated level generation provides a strong benefit to the system as we can build on this to create a set of proscribed environments for various L2M objectives (e.g. transfer, adaptation, curriculum's, goal-driven perception...).

## Setup

### Source
Follow the instructions on the [GVGAI_GYM](https://github.com/rubenrtorrado/GVGAI_GYM) repository.

### Docker Registry (HRL Only)
All shared images, including the images for development and CI testing, reside in [Harbor](https://harbor.hrl.com) under `darpa_l2m`. To use the images, you must configure your docker daemon to support insecure registries.
**Mac**
- Click the docker whale in the top toolbar 
- Select `Preferences`->`Daemon`
- Click `+` for Insecure Registries
- Add harbor.hrl.com:5000
- Click `Apply and Restart`

Once your daemon is configured properly, you will need to authenticate against harbor with your LDAP credentials
`docker login harbor.hrl.com:5000`

To pull images, run
`docker pull harbor.hrl.com:5000/darpa_l2m/<image_name>`


### Dockerfile
The Dockerfile setups GVGAI properly for development use. It is packaged with a virtual X-session with VNC, Juypter session, etc. You can build the image in the directory with the Dockerfile with:
```bash
docker build -t gvgai .
```
which will create a new docker image with the name `gvgai` on your system. You can verify its addition by running `docker image ls` or `docker images`.


## Usage
Once the docker image is installed, a container needs to be started. Two approaches have been provided: standard docker CLI and docker-compose. For both, the user will need to modify some components for the container to start properly (notably: volumes, ports, etc.). The `docker-compose.yml` file contains the configuration to be modified for the docker-compose technique.

To run a container using docker-compose (preferred):
```bash
docker-compose up
```
which will create a container with the specifications outlined in `docker-compose.yml`

To run a container using CLI:
```bash
docker run --name gvgai -v $(pwd):/root/code/host" -p 8889:8888 -p 6007:6006 -p 5910:5900 gvgai
```
which will create a container called 'gvgai' with the current working directory mounted at '/root/code/host/' within the container.  Then you always go back and restart this container using a `docker run gvgai` command as long as the container has not been removed.


## Example
A simple tutorial video is available [here](https://www.youtube.com/watch?v = O84KgRt6AJI&feature = youtu.be), however note that some of the code examples in the video are now out of date.  A quick test to verify things have been installed correctly:
```python
import gym
import gym_gvgai
envs = [env.id for env in gym.envs.registry.all()]
[i for i in envs if 'gvgai' in i]
```
Here `envs` lists all available Gym environments, while all the gvgai specific environments should be printed to the screen.  In general the naming format for a given gvgai game is: `name-lvlN-vM`.  For example:
```python
import gym
import gym_gvgai
env = gym.make('zelda-lvl0-v0')
env.reset()
for i in range(1000):
	obs,reward,done,info = env.step(env.action_space.sample())
	env.render()
	if done:
		env.reset()
```
will open the first level in the Zelda game, and have an agent perform random actions for 1000 timesteps resetting the environment as needed.  If you have an active display on your setup, you should see a window appear rendering the environment.

## Extension (games/levels)
New games/levels can be added to the existing set by looking in the GVGAI_GYM code at `GVGAI_GYM/gym_gvgai/envs/games`. The directory structure follows a sub-directory per game (e.g. Zelda):
 ```bash
GVGAI_GYM/gym_gvgai/envs/games# ls zelda_v0/
 zelda.txt  zelda_lvl0.txt  zelda_lvl1.txt  zelda_lvl2.txt  zelda_lvl3.txt  zelda_lvl4.txt
 ```
Here `zelda.txt` contains the game definition file, as described in the [Video Game Definition Language](https://github.com/EssexUniversityMCTS/gvgai/wiki/VGDL-Language) that GVGAI is based on.  You can copy this text to a new folder, e.g. zelda-v1, to make small changes, including core functionality as well visual appearance.

Similarly, the `zelda-lvl*.txt` files define the various levels that can be played using the rules in the `zelda.txt` definitions file. You can make alterations to these files to iterate changes to the various tasks (here the nomenclature related to L2M would be **game ==  environment**, **task ==  level**).

To provide some intuition here you can see below the mapping in the Zelda game between characters and sprites and the ascii txt file defining the first level.  To create a new level, you would simple create a new txt file using the defined characters in the desired configuration to construct the new task.

```
LevelMapping
  g > floor goal
  + > floor key
  A > floor nokey
  1 > floor monsterQuick
  2 > floor monsterNormal
  3 > floor monsterSlow
  w > wall
  . > floor

#level definition:

wwwwwwwwwwwww
wA.......w..w
w..w........w
w...w...w.+ww
www.w2..wwwww
w.......w.g.w
w.2.........w
w.....2.....w
wwwwwwwwwwwww
 ```

Currently there seems to be a limit of registering 5 levels for any given game with the GYM_GVGAI framework.  Risi's lab worked around this by creating some code to dynamically access levels on demand.  A hacky way of accomplishing this would simple be to swap text files in and out as needed, although this has not been tested yet. Eventually we will have to adopt a systematic approach similar to Risi if we want to take full advantage of the level generation functionality in the GVGAI framework.

## Concerns

- As of now, there is no easy way to generate a non-visual state-space representation for a given game.
- I don't see an easy way to render in subjective/first-person mode.
- I think all the currently available games (at least for OpenAI Gym) are in discreet (non-continuous) action spaces.  GVGAI does support continuous physics, and some example games are listed in the gvgai_gym repo at: `GVGAI_GYM/gym_gvgai/envs/gvgai/examples/contphysics`.  Might be relatively easy to integrate these with Gym?
- All of GVGAI is based on Java code, which the HRL is particularly weak on.  Other partners, e.g. Risi, seem to have good footing in this language, and @anpatel at HRL can assist, however changes to the gvgai framework, and/or sofphisticated development of automatic level generators will be tricky.

## TODO
* Currently working on a general framework for integrating level generation into our work-flow, this will be inspired by Risi's work but paired down for our specific needs. 
* Hopefully we can develop/exploit hooks into the java code so we can extract non-pixel based state-space representations from Gym calls, e.g env.step().
