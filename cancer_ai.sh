#run bash commands
pip install virtualenv --user
python -m virtualenv --python=python venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py
deactivate
