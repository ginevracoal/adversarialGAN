DEVICE="cuda"
ODE_IDX=1
TRAINING_STEPS=10
DIR="../experiments/cartpole"
FOURPLOTS=True
SCATTER=False
HIST=False
DARK=False

source ../venv/bin/activate

python train_cartpole.py --device=$DEVICE --ode_idx=$ODE_IDX --training_steps=$TRAINING_STEPS --dir=$DIR
python tester_cartpole.py --device=$DEVICE --ode_idx=$ODE_IDX --dir=$DIR
python plotter_cartpole.py --ode_idx=$ODE_IDX --dir=$DIR --fourplots="$FOURPLOTS" --scatter="$SCATTER" --hist="$HIST" --dark="$DARK"

deactivate