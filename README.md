# Plyplayer
![Alt text](fig1.png?raw=true "plyplayer")

## Installation
Python 3.8 <br />
pip install -r requirements.txt

## Download MANO model and Manopth
- Go to the mano website and download the model.
- Clone manopth (https://github.com/hassony2/manopth) and copy manopth folder (inside) into the plyplayer folder.
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

## Example command
```
python h2oplayer.py --source /media/taein/storage/h2o_v5/subject1/h1/0/ --start 10 --end 15 --hand_proj --hands --object --ego --rgb --ply_create --action_info
```
## How to control
space : stop/play <br />
,: previous frame <br />
.: next frame <br />
q: toggle for checking failure frames <br />
u: update poses <br />
r: return to the first frame <br />

ctrl + drag: Move camera position <br />
wheel: zoom-in/out  <br />
