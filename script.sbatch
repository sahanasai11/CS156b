#!/bin/bash 

# Name of jobs in SLURM queue 
#SBATCH --job-name=Run_Model_Sbatch_job 

# Output and error log locations (captures stdout and stderr)
#SBATCH --output=/home/hsiaveli/%j.out
#SBATCH --error=/home/hsiaveli/%j.err

# Account for Caltech to charge for using hpc
#SBATCH -A CS156b

# Estimated time for the job to run 
#SBATCH -t 1:30:00

# Total numer of concurrent srun tasks. 
#SBATCH --ntasks=1

# Number of CPU threads for each task listed above 
#SBATCH --cpus-per-task=1

# Total amount of system RAM for all tasks. Specifit units with M and G 
#SBATCH --mem=32G

# Request a single Tesla P100 GPU
#SBATCH --gres=gpu:1

#send status emails to a single email
#SBATCH --mail-user=hsiaveli@caltech.edu

# Enable email notifications for changes to the job state 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load any modules you may need 
module load python3-3.8.5

# Setup a python enviroment 
# CHANGE TO WHOEVER IS RUNNING THE SCRIPT
source /home/hsiaveli/vens/cs156b/bin/activate

# Anything else
