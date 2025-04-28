# Learning-Multi-Body-Dynamics
## Simulator-Augmented Interaction Networks (SAIN) for Proxy Pushing  
Learn and model the dynamics of an agent pushing a target object, and use it to plan a control strategy to make the target object reach the target position   

## Motivation
In some tasks, we are not able to directly manipulate the desired object but we can use other objects as a proxy to interact with the desired object. Here, the proposed extension experimental setup is similar to [Ajay et al., ICRA, 2019], where we push an object to the goal by using an intermediate object.   
The robot will push this intermediate object into contact with the target object, and plan for a sequence of actions such that the interaction between the two will drive the target object to its goal.  

---
**SAIN Demos**  
![Image](https://github.com/user-attachments/assets/20b7ea1d-b6cd-4891-a6a8-8370b0537e3c)

![Image](https://github.com/user-attachments/assets/6ff2f3f6-bee2-40b3-97bf-924ed00021b8)

![Image](https://github.com/user-attachments/assets/cdb7b1ff-7b4d-46bb-a48f-b3432ff288dc)

![Image](https://github.com/user-attachments/assets/30182c2f-31c8-4505-9554-6eff1739ee0e)

![Image](https://github.com/user-attachments/assets/8b9bf545-34ea-4391-a9c1-70cb3e327ff8)

---
> **Task**: Use a Franka Panda arm to push a **middle object**  as a proxy tool to indirectly drive a **target object**  into a green goal region.    
> **Approach**: Re-implement Ajay *et al.* ICRA 2019’s **Simulator-Augmented Interaction Network (SAIN)** combined with **MPPI** for closed-loop control in PyBullet.  



## 📂 Repository Structure  
├── assets/ # URDF models and geometry  
├── dataset/ # Collected .npz data chunks   
├── checkpoints/ # Trained model weights   
├── data_collected.py # Multi-process data collection script  
├── data_processing.py # Dataloader   
├── demo.py # SAIN + MPPI closed-loop demo (pybullet GUI)   
├── run_mppi.py # SAIN + MPPI closed-loop demo (produces GIF)   
├── mppi.py # Generic MPPI sampler   
├── mppi_control.py # SAIN + MPPI closed-loop demo (produces GIF)   
├── panda_pushing_env.py # Two-disk PyBullet environment   
├── sain_model.py # Interaction Network & SAIN implementations   
├── visualizers.py # get gif sample  
└── README.md # ← You are here  

## Environment Setup
```
conda create -n sain_torch python=3.10  
conda activate sain_torch   
chmod +x install.sh  
./install.sh  
python demo.py
```

## Future Work
> Graph-based ODE for continuous-time prediction  
> Extend to 3D proxy pushing with obstacles  
> Deploy on real Franka Panda with vision feedback  

## References
Ajay A., Bauza M., Wu J., Fazeli N., Tenenbaum J.B., Rodriguez A., Kaelbling L.P.“Combining Physical Simulators and Object-Based Networks for Control.” ICRA 2019 arXiv:1904.07301  
