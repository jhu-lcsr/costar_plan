# Download the grasp dataset with xonsh shell:
# cd /path/to/grasp_dataset.py
# source xonsh_download.xsh
import sys
sys.path.insert(0, '')
import grasp_dataset
g = grasp_dataset.GraspDataset(dataset='all')
g.download()