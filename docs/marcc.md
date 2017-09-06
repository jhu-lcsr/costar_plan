# MARCC

If you are a Johns Hopkins student, you may have access to MARCC. This doc is a work in progress showing ways you can use CTP with MARCC and in particular with SLURM.

## Interactive Session

Create an interactive session with a GPU to make sure your code and extensions all work. This limits you to about an hour of usage, so it is not a permanent solution.
```
interact -n 6  -p gpu -g 1
```

## Training

Run a training script with `sbatch`:
```
sbatch -n 6 -p unlimited -g 1 --time=24:0:0 script.sh
```

Scripts are contained in `costar_plan/slurm`. `#SBATCH` directives should be listed immediately, e.g.:
```
#!/bin/bash -l
#SBATCH --job-name=b500v2
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH -p unlimited
#SBATCH -g 1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=cpaxton3@jhu.edu
```
from [the blocks500v2 example script](../slurm/blocks500v2.sh).
