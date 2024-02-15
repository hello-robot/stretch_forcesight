# Stretch ForceSight

**IMPORTANT: When moving the robot based on the output of ForceSight you must be ready to push the run-stop button and terminate the code. The robot will take actions that put itself, its surroundings, and people at risk!**

This is a repository for running a pretrained ForceSight model with D405 images and a Stretch 3 mobile manipulator from Hello Robot. **This model was trained on an older Stretch robot with a significantly different camera configuration and a wrist-mounted force-torque sensor. To achieve performance comparable to the original system, the model would need to be finetuned or retrained for Stretch 3.** 

In spite of the significant limitations associated with running this model on Stretch 3, we believe it is functional enough at approaching and grasping objects to be informative, especially with respect to the ability of ForceSight to generalize to new embodiments and objects outside of the original robot training data. 

The following GIFs show an example where a Stretch 3 running this code successfully approached and grasped a translucent cup on a glass-topped surface. **Please note that the lift action was a hand-coded hack executed when the system detected that the grasp action primitive was successful. The hack has been removed from the code.** **In general, failure is common, so be prepared!**

| Stretch Moving to ForceSight Goals | Visualization of ForceSight Goals |
| -------------------------- | ---------------------- |
| ![](/gifs/forcesight_cup.gif) | ![](/gifs/forcesight_cup_stretch_view.gif) |

If you would like to modify, train, or test a ForceSight model or learn more about ForceSight, you should go to the following two websites. **We would welcome models trained for Stretch 3!**

[https://github.com/force-sight/forcesight](https://github.com/force-sight/forcesight)

[https://force-sight.github.io/](https://force-sight.github.io/)

## Recommended Installation

This installation has been tested with Ubuntu 22.04.

First, clone this repository on the robot and an external computer that you plan to run the deep model. The external computer should have a powerful GPU to achieve a high-enough rate to support closed-loop control. For example, in our testing we used a desktop with an NVIDIA GeForce RTX 4090 GPU and good WiFi connectivity to achieve 15 Hz, which is the frame rate of the D405 camera on Stretch 3. 

```
git clone https://github.com/hello-robot/stretch_forcesight/
```
### Configure Your Network

Next, edit `forcesight_networking.py` on your robot and your external computer to match your network configuration. In particular, you need to specify the robot's IP address and the external computer's IP address.

### External Computer Installation

#### Download the Pretrained Model

To use this repository, you will need to download a pretrained ForceSight model into the external computer's repository directory. The code has only been tested with `model_best.pth`. As of February 2, 2024, you can download this 1.29 GB model from [this OneDrive folder](https://onedrive.live.com/?authkey=%21ALvdUAiUg4s8LPY&id=79F9A071FA899B37%2179715&cid=79F9A071FA899B37).

#### Create a Virtual Environment

Next, create a Python virtual environment in the repository. You can learn about Python virtual environments from [this guide at python.org](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Examples of commands to run follow: 

```
cd ./stretch_forcesight
sudo apt install python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
which python
```

Now install the required Python packages in the virtual environment you created.

```
pip3 install --upgrade pip
python3 -m pip --version
pip3 install -r forcesight_min_requirements.txt
pip3 install pyzmq
pip3 install urchin
```

### Robot Installation

You need to characterize your robot's gripper, so that the code can estimate grip force. Run the following code on your robot. 

```
cd ./stretch_forcesight
python3 characterize_gripper.py
```

## Running ForceSight

First run the server that sends D405 images from the the robot by running the following code on your robot. 

```
python3 send_d405_images.py -r
```

Next, go to the repository's directory on the external computer and activate the virtual environment you set up. 

```
cd ./stretch_forcesight
source .venv/bin/activate
```

Run the following client on the external computer to receive and process the images from the robot's D405. The code can take significant time to run as it loads the model parameters, especially the first time.

```
python3 recv_and_forcesight_d405_images.py 
```

You should now be seeing visualizations of the estimated fingertip poses on the external computer. Initially, it starts in "Do Nothing" mode and no text prompt is provided to ForceSight.

To specify a text prompt for ForceSight,  open up another terminal on the external computer and run the following code.

```
python3 auto_prompt.py
```

Auto prompt can append action primitives automatically when the system detects that the previous action has succeeded. The system can detect when the approach action has succeeded and start the grasp action. Beyond this, automatic detection of completed actions does not work well enough on Stretch 3 to use. 

We recommend that you start by providing the full prompt with an action primitive instead of having the system add the action primitives. ForceSight is also sensitive to the prompts with which it was trained. For example, you can try the following prompts one at a time.

```
pick up the cup, approach
pick up the cup, grasp
```

After specifying a prompt that applies to an object currently in the D405's field of view, you should see visualizations of the ForceSight goals on the external computer. They should look similar to the "Visualization of ForceSight Goals" GIF above.

Other prompts and objects can be found at the official ForceSight websites. **Prompts other than "pick up the OBJECT_NAME" with the "approach" and "grasp" action primitives have not been tested or are known to perform poorly.**

[https://github.com/force-sight/forcesight](https://github.com/force-sight/forcesight)

[https://force-sight.github.io/](https://force-sight.github.io/)

## Moving the Robot to ForceSight Goals

You can now have your Stretch robot move to the goals provided by ForceSight using Cartesian visual servoing with velocity control. To do so, run the following command on the robot.

**IMPORTANT: This code uses an unmodified deep model trained with a significantly different robot as part of an academic research project.  When allowing the robot to move based on ForceSight you must be ready to push the run-stop button and terminate the code. The robot will take actions that put itself, its surroundings, and people at risk! Be careful! USE AT YOUR OWN RISK!**

```
python3 forcesight_servoing.py -r
```

## Cite ForceSight

ForceSight is the result of academic research. As such, citations to [the academic paper](https://arxiv.org/abs/2309.12312) are a great way to recognize the work and reward the team. To cite ForceSight, you can use the following BibTeX entry or text.

*Jeremy A. Collins and Cody Houff and You Liang Tan and Charles C. Kemp. [ForceSight: Text-Guided Mobile Manipulation with Visual-Force Goals](https://arxiv.org/abs/2309.12312). Accepted to the IEEE International Conference on Robotics and Automation (ICRA), 2024.*

```
@InProceedings{collins2023forcesight,
      title={ForceSight: Text-Guided Mobile Manipulation with Visual-Force Goals}, 
      author={Jeremy A. Collins and Cody Houff and You Liang Tan and Charles C. Kemp},
      booktitle={Accepted to the IEEE International Conference on Robotics and Automation (ICRA)},  
      year={2024},
}
```
