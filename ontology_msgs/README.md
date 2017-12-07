# ontology_msgs

This project contains ROS message definitions used by [onto_agent](https://gitlab.ics.ei.tum.de/mix-reality/vr_recognition/onto_agent) and related components.

## Usage:

Clone this repo into your ROS workspace. For the easiest setup, keep this project [ontology_svcs](https://gitlab.ics.ei.tum.de/mix-reality/vr_recognition/ontology_svcs) and [onto_agent](https://gitlab.ics.ei.tum.de/mix-reality/vr_recognition/onto_agent) in the same ROS workspace (but as long as these packages can reference each other, everything should be fine even if they're located in separate workspaces).

## Message Definitions

#### Activity
```slim
string name
string className
string[] inHand
string[] actedOn
```

Basic activity definition

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `name`          | Name of activity (must be unique) |
| `className`     | Ontology class for the activity |
| `inHand`        | List of ontology classes which can be used for the `inHand` property of this activity |
| `actedOn`       | List of ontology classes which can be used for the `actedOn` property of this activity |

---

#### DisplayLabel
```slim
string text
uint32 lbl_id
```

Used to send information from `onto_agent` to labels in the Unity at run-time. Useful for debugging and showing data like the current hand classification state in a manner visible to users in VR, so they don't have to constantly switch between looking at the simulation an looking at `onto_agent`'s terminal output.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `text`          | Text to dislay |
| `lbl_id`        | Unique id of the label in Unity to apply the text to (necessary since there may be many such labels for different output) |

---

#### Graph
```slim
GraphNode[] nodes
```

Message for passing directed graphs between ROS nodes

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `nodes`         | List of all nodes and related edges in the graph |

---

#### GraphNode
```slim
string name
string[] neighbors
float32[] weights
```

A single node and connected edges in a directed graph.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `name`          | Node name/id. Must be unique in the graph. |
| `neighbors`     | List of names/ids this node has outgoing edges to |
| `weights`       | Corresponding weights for outgoing edges. Must be same length as `neighbors` |

---

#### GraphUpdate
```slim
int32 ADD=1
int32 REMOVE=2
int32 CHANGE=3

int32 operation
GraphNode node
```

Used to provide incremental update information about graphs. This allows us to provide live updates for large graphs without having to constantly send the entire graph after every update, but it DOES require that both the sender and receiver have been in communication from the start.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `operation`     | **ADD** - New graph data has been inserted <br />**REMOVE** - The given graph data has been removed <br />**CHANGE** - The given graph data has been updated |
| `node`          | Node and linked edge data to update |

---

#### HandUpdate
```slim
ObjUpdate handState
float32 grasp
float32 use
string obj_inHand
```

Sent any time an observed hand may change state. When connected to Unity, these are sent as often as possible, but *at least* as often as once every log interval for each hand. When connected to a physical robot, they should be sent as often as tf provides new pose information.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `handState`     | Current pose & velocity of the hand |
| `grasp`         | Value between 0 and 1 representing the grasp state. <br /> 0 - Hand is fully open <br /> 1 - Hand is fully closed |
| `use`           | Value between 0 and 1 representing "use" state (e.g. squeezing held object, pulling trigger, pushing button on held device, etc). <br /> 0 - "Use" action is not happening <br /> 1 - "Use" action is happening |
| `obj_inHand`    | Instance name of object currently held in this hand, or "NONE" if hand is empty |

---

#### ObjUpdate
```slim
time timestamp
string name
geometry_msgs/Vector3 position
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 velocity
```

Sent any time an object may change state. When connected to Unity, one ObjUpdate is sent for each recorded object on every log interval. Other systems may only send updates when objects are confirmed to have moved.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `timestamp`     | Current time when the update happened |
| `name`          | Object name (in Unity this is the name of the GameObject) |
| `position`      | Current 3D position of object |
| `orientation`   | Quaternion representing current rotation of object |
| `velocity`      | Current linear velocity of object |

---

#### OntoPropertyList
```slim
OntoProperty[] properties
```

A list of object properties

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `properties`    | List of object instance properties |

---

#### OntoProperty
```slim
string name
string value
```

Simple key/value pair for object properties (e.g. 'wetness' : 'dry')

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `name`          | Property name |
| `value`         | Property value |

---

#### PropertyChanged
```slim
time timestamp
string instanceName
string propertyName
string propertyValue
```

Sent when an object instance undergoes a property change (e.g. 'wetness' : 'dry' -> 'wet')

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `timestamp`     | Time at which property change happened |
| `instanceName`  | Object name (e.g. name of GameObject in Unity) |
| `propertyName`  | Name of the property which has changed (e.g. 'wetness') |
| `propertyValue` | New value of the property (e.g. 'dry') |

---

#### SimulationStateChange
```slim
uint32 SIMULATION_LOADED  = 1
uint32 SIMULATION_STARTED = 2
uint32 SIMULATION_STOPPED = 4

uint32 new_state
time timestamp
```

Used to notify `onto_agent` when the simulation starts/stops. This allows `onto_agent` to maintain a continuous time log without any discontinuities when the simulation is paused, resumed, between multiple separate recording sessions, etc.

| **_Parameter_** | **_Meaning_** |
| --------------- |:------------- |
| `new_state`     | **SIMULATION_LOADED** - Sent when simulation loads a new scene <br />**SIMULATION_STARTED** - Sent when recording begins <br />**SIMULATION_STOPPED** - Sent when recording ends |
| `timestamp`     | The current time in the simulation |

