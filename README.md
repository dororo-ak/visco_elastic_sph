# Particle based simulation implementation - Adama KOÏTA
## Three different Example of fluid behaviour with different parameters
<p align="center">
  <img src="images/veryGluant.gif" alt="sticky behaviour" width="35%" />
  <img src="images/semiliquide.gif" alt="Description de l'image 2" width="35%" />
  <img src="images/anotherFlow.gif" alt="Description de l'image 3" width="35%" />
  <img src="images/liquid.gif" alt="Description de l'image 3" width="35%" />
</p>

## Drop formation thanks to double density

<p align="center">
  <img src="images/doubleDensityOneDropAddedMouse.gif" alt="sticky behaviour" width="40%" />
</p>

## Comparison of viscous SPH simulation and Viscous particle based viscoelastic simulation

<p align="center">
  <img src="images/veryGluant.gif" alt="sticky behaviour" width="40%" />
  <img src="images/instabilitiesSPH.gif" alt="sph viscous" width="40%" />
</p>

## ImGui  interface
![imgui](images/imguiInterface.png)




- To change the view angle if you are in "Add particle with mouse" mode, click on **CTRL** and move the mouse.
- if you put too high parameters the simuation will explode and the program will stop. I advise to not touch too much to beta and to not change timestep dt. Do not increase too much k and k_near 
-If you check the simulation SPH only, make sure to have a high k.

### Not block template particles checked - only 1 particle
<p align="center">
  <img src="images/template.png" alt="template" width="40%" />
</p>


#### Not block template particles unchecked - A template of 1024 particles
<p align="center">
  <img src="images/notTemplate.png" alt="no Template" width="40%" />
</p>

## Code definition 

### OpenMP
 if you're computer can't manage openMP, you can directly comment the line "PARALLEL_CPU_VERSION" at the top of the main code
![openMp](images/codeBeginning.png)


NB : Before the simulation run the Makefile