# Third-party libraries
from datetime import date, datetime, timedelta
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
# import ariel.simulation.tasks.gate_learning
from ariel.simulation.tasks.gate_learning import xy_displacement, y_speed

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []
FITNESS_HISTORY = []
TIME_HISTORY = []


def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def controller(model, data, to_track):

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    w1 = np.random.randn(input_size, hidden_size) * 0.1
    w2 = np.random.randn(hidden_size, hidden_size) * 0.1
    w3 = np.random.randn(hidden_size, output_size) * 0.1

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = sigmoid(np.dot(inputs, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outputs = sigmoid(np.dot(layer2, w3))

    # Scale outputs to [-pi/2, pi/2]
    data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

def controller_2(model, data, to_track):
    # print(data.qpos)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    fitness = 0
    if len(HISTORY) > 1:
        pos =  np.array(HISTORY)

        # More absolute displacement = Better
        fitness = xy_displacement(pos[-1], pos[-2]) # example, higher values = higher fitness
        
        # Higher speed on Y axis = better
        fitness = max(y_speed(pos[-1], pos[-2], (TIME_HISTORY[-1] - TIME_HISTORY[-2]).total_seconds()), 0.0)


    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu


    # Initialize the networks weights randomly
    w1 = np.random.randn(input_size, hidden_size) * 0.1
    w2 = np.random.randn(hidden_size, hidden_size) * 0.1
    w3 = np.random.randn(hidden_size, output_size) * 1.2
    # print(w3)

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = sigmoid(np.dot(inputs, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outputs = sigmoid(np.dot(layer2, w3))

    # Scale outputs to [-pi/2, pi/2]
    data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())
    FITNESS_HISTORY.append(fitness)
    TIME_HISTORY.append(datetime.now())

def sine_controller(model, data, to_track):
    if  PARAMS is None:
        raise ValueError("No parameters set for controller!")

    # simulation time
    t = data.time
    num_joints = model.nu
    print(num_joints)

    # apply sine waves to each joint
    for j in range(num_joints):
        A, w, p = PARAMS[j]
        data.ctrl[j] = A * np.sin(w * t + p)

    # Save movement to history (track gecko core position)
    HISTORY.append(to_track[0].xpos.copy())

def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Set the control callback function
    # This is called every time step to get the next action.
    # Testing the sine_controller function
    global PARAMS, START_TIME
    PARAMS = [
        (1.0, 2.0, 0.0),   # joint 0
        (0.5, 1.5, 0.2),   # joint 1
        (2.0, 0.8, 0.5),   # joint 2
        (1.2, 1.0, 0.1),   # joint 3
        (0.8, 2.2, 0.3),   # joint 4
        (1.5, 1.7, 0.4),   # joint 5
        (2.5, 0.9, 0.6),   # joint 6
        (1.1, 1.4, 0.7),   # joint 7
    ]

    mujoco.set_mjcb_control(lambda m,d: controller_2(m, d, to_track)) # random_move(m, d, to_track)

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )
    show_qpos_history(HISTORY)
    print('max: ', np.max(FITNESS_HISTORY))
    print('mean: ', np.mean(FITNESS_HISTORY))
    print('min: ', np.min(FITNESS_HISTORY))
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()


