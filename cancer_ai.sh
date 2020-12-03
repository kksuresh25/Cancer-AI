#run bash commands
pip install virtualenv --user
python -m virtualenv --python=python2 venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py
deactivate
