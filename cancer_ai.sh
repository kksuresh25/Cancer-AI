#!/bin/bash

#run bash commands
pip2 install virtualenv --user
python2 -m virtualenv cancerai_venv
source cancerai_venv/bin/activate
pip install -r requirements.txt
python predict.py
deactivate
