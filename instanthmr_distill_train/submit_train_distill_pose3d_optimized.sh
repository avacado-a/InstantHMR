#!/bin/sh -e

echo "example: $0 --gpu 1"

# if `requirements.txt` is present, virtual environment is installed and used
EXEC="python3 train_distill_pose3d_optimized.py"

GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers5.sh

# --data_root points at the dataset synced by datasynch_perso (see below).
# It is synced next to this project folder, i.e. ../instanthmr_data once we
# cd into instanthmr_distill_train.
EXEC="python3 -u train_distill_pose3d_optimized.py --data_root /datasets/instanthmr_data --num_workers 8 --batch_size 128 $useropt"

# SBATCH --output/--error paths are relative to the submission directory, so
# create the log folder before submitting.
mkdir -p instanthmr_distill

jobMessage=$(
######################### SBATCH launcher
######################### #SBATCH --option # <= this is an enabled paramater
######################### ##SBATCH --option # <= this is a disabled parameter
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="instanthmr_distill"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=instanthmr_distill/stdout.txt
#SBATCH --error=instanthmr_distill/stderr.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=35000
#SBATCH --time=4-00:00:00

. /pfcalcul/tools/sbatchHelpers5.sh

/pfcalcul/tools/datasync /datasets/instanthmr_data


time $EXEC

eof
)

showSubmitted
