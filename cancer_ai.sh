#run bash commands
pip2 install virtualenv --user
virtualenv --python=python2 venv
source venv/bin/activate
pip install -r requirements.txt --user
python predict.py
