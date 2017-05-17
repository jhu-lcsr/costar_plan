
# Development Notes

These notes include practices for CoSTAR in general and for CoSTAR PLAN, its learning and planning module, in particular.

***Philosophy:*** Largely, we subscribe to the "single repository" school of software development. All dependencies are inside the main CoSTAR plan repository, wherever this is practical. Individual components may be spun out as separate packages, assuming there's any demand.

We use **PEP8** formatting where ever possible, 80 chars per line. Run:
```
autopep8 -ri
```
on your files and make sure they look nice before adding them.

## CoSTAR Component

All CoSTAR classes that represent components should ideally extend the `CostarComponent` class in `costar_stack/costar_component`.

For example:
```
from costar_component import CostarComponent

class MyComponent(CostarComponent):
  def __init__(self, myargs, *args, **kwargs):
    super(MyComponent, self).__init__(*args, **kwargs)
```

See the [example component code](costar_component/example_component.py).

## Arm Configurations

Arm configuration should go in `costar_stack/costar_bringup/launch/config`. For example, the [UR5 config](costar_bringup/launch/config/ur5.launch) looks like this:
```
<launch>
  <!-- Smartmove params -->
  <group ns="costar">
    <rosparam param="home">[0.30, -1.33, -1.80, -0.27, 1.50, 1.60]</rosparam>
    <group ns="robot">
      dof: 6
      base_link: base_link
      end_link: ee_frame
    </group>
    <group ns="smartmove">
      <rosparam param="available_objects">
        - sander_makita
        - node_uniform
        - link_uniform
        - drill
        - table
      </rosparam>
      <rosparam param="regions">
        - in_back_of
        - in_front_of
        - right_of
        - left_of
        - up_from
        - down_from
      </rosparam>
      <rosparam param="references">
        - world
      </rosparam>
    </group>
  </group>
</launch>
```

The alternative is to write a yaml file.

## Tests

Tests go in `costar_stack/costar_bringup/tests`. Right now these include:
  - [IIWA sim test](costar_bringup/tests/iiwa_test.py): launches gazebo, servos to several positions and verifies that the robot makes it.


Make sure the tests work before making any changes.

### Travis CI

  - IIWA sim test is not included in Travis.
