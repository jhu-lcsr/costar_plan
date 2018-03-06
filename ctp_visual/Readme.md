
# Visual Robot Task Planning

## To Run:

You can use all the usual `ctp_model_tool` options to specify model parameters to some extent:
```
rosrun ctp_visual gui_run.py --model_directory $HOME/.costar/models
```

Note that it will load the `conditional_image` architecture by default, with `--features multi`. Otherwise this will not work.

## Other Helpful Commands

### Create Videos

```
rosrun costar_models make_video --data_file data.h5f \
  --model_directory ~/.costar/models \
  --model conditional_image --features multi --cpu
```

### Generate Parallel Predictions

```
rosrun costar_models make_parallel_predictions --data_file data.h5f \
  --model_directory ~/.costar/models --model conditional_image \
  --features multi --cpu
```
