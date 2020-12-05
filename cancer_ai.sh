#!/bin/bash

#run bash commands
pip install virtualenv --user
python3 -m venv cancerai_venv
source cancerai_venv/bin/activate
pip install -r requirements.txt
python predict.py
deactivate
