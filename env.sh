module purge
module load gcc/11.4.0 openmpi/4.1.4 python/.3.11.4
#export PYTHONPATH=$(pwd):$PYTHONPATH

if [ ! -d "$ENV" ]; then
    python -m venv ENV
fi

source ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
