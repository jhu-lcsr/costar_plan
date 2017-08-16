# Docker integration with costar_plan

To run costar_plan from a docker container please install nvidia-docker.
nvidia-docker is a wrapper over docker that allows you to leverage your nvidia graphics card over docker easily.
Instructions can be found [here](https://github.com/NVIDIA/nvidia-docker).

After installing run the following command:
```
sudo nvidia-docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix alee156/nvidia-costar
```
