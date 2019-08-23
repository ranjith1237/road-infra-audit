#!/bin/bash                                                               
#SBATCH --job-name=bash             # Job name                        
#SBATCH --time=23:00:00                 # Time limit hrs:min:sec          
#SBATCH --partition=cosmos            # Mention partition-name. default 
#SBATCH --output=logs_2019-07-27-17-01-07.out        # Output file.                    
#SBATCH --gres=gpu:1    # N number of GPU devices.            
#SBATCH --qos=ninja
#SBATCH --mail-type=ALL                 # Enable email                    
#SBATCH --mail-user=ranjithreddy1061995@gmail.com    # Where to send mail  
#SBATCH --mem=32G                     # Enter memory, default is 100M. 

srun python detect.py