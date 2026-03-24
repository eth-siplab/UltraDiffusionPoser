# Ultra Diffusion Poser: Diffusion-Based Human Motion Tracking from Sparse Inertial Sensors and Ranging-based Between-sensor Distances (CVPR 2026)

[Dominik Hollidt](), [Tommaso Bendinelli](), [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>

Abstract
----------
Methods using inertial measurement units (IMUs) provide a wearable alternative to camera-based motion capture.
To mitigate drift from inertial signals, recent sparse inertial pose estimators integrate inter-sensor distances measured by ultra-wideband (UWB) ranging. 
So far, UWB distances have only been used as an additional input feature, ignoring the physical constraints they impose on sensor positions.
However, these distances can also be used to reconstruct the underlying 3D sensor layout, which in turn provides more informative input for pose reconstruction.
We propose Ultra Diffusion Poser, a diffusion model that explicitly models these geometric constraints.
It includes a Spatial Layout Module that analytically reconstructs the 3D sensor positions from UWB measurements.
These sensor positions are used alongside IMU signals and UWB distances as a conditioning signal during diffusion.
Still, network predictions can violate inter-sensor distance measurements.
To address this, we introduce UWB-Diffusion Guidance, which encourages alignment between predicted poses and measured distances during diffusion sampling.
Together, these contributions enable our model to achieve state-of-the-art performance, reducing joint position error by up to 22\% over prior work.

### Code
coming soon
