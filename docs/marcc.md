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
```
