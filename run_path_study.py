import os
from time import sleep


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

nb_seeds = 5
values = [0.002]
paths = [2, 3, 4]

for i in range(nb_seeds):
    for value in values:
        for p in paths:
            job_file = os.path.join(job_directory, "path_study_{}%.slurm".format(p))

            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --account=oke@cpu\n")
                fh.writelines("#SBATCH --job-name=path_study_{}\n".format(p))
                fh.writelines("#SBATCH --partition=cpu_p1\n")
                fh.writelines("#SBATCH --qos=qos_cpu-t4\n")
                fh.writelines("#SBATCH --output=path_study_{}%_%j.out\n".format(p))
                fh.writelines("#SBATCH --error=path_study_{}%_%j.out\n".format(p))
                fh.writelines("#SBATCH --time=40:00:00\n")
                fh.writelines("#SBATCH --nodes=1\n")
                fh.writelines("#SBATCH --ntasks=24\n")
                fh.writelines("#SBATCH --hint=nomultithread\n")

                fh.writelines("module load pytorch-cpu/py3/1.4.0\n")

                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")

                fh.writelines(
                    "srun python -u -B train.py --max-path-len {} --intervention-prob {} --save-dir 'path_study_{}_{}%/' 2>&1 ".format(p, value, p, value))

            os.system("sbatch %s" % job_file)
            sleep(1)