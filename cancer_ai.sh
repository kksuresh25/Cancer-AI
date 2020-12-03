#run bash commands
pip2 install virtualenv --user
python2 -m virtualenv --python=python2 venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py
deactivate
