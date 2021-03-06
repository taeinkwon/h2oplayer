# H2OPlayer
![Alt text](fig1.png?raw=true "H2OPlayer")

## Installation
- Python 3.8
- ```git clone https://github.com/taeinkwon/h2oplayer```
- ```pip install -r requirements.txt```

## Download MANO model and Manopth
In this app, we use [MANO](https://mano.is.tue.mpg.de/) model from MPI and some part of [Yana](https://hassony2.github.io/)'s code. Note that all code and data you download follow the licenses of each repository.
- Go to the [mano website](https://mano.is.tue.mpg.de/) and download models and code.
- Copy the ```mano``` folder into ```h2oplayer``` folder
- Copy the ```model``` folder into ```h2oplayer/mano``` folder
- Clone [manopth](https://github.com/hassony2/manopth) and copy ```manopth``` folder (inside) into the plyplayer folder.
- Folder structure should look like the following:
```
h2oplayer/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
  manopth/
    argutils.py
    demo.py
    ...
```

## Demo
```
python h2oplayer.py --source /media/taein/storage/h2o_v5/subject1/h1/0/ --start 10 --end 15 --hand_proj --hands --object --ego --rgb --ply_create --action_info
```
## How to control
space : stop/play <br />
,: previous frame <br />
.: next frame <br />
r: return to the first frame <br />

ctrl + drag: Move camera position <br />
wheel: zoom-in/out  <br />
