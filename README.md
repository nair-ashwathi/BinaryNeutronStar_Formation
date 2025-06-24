# BinaryNeutronStar_Formation
This contain all the post-processing codes used to simulate the helium-neutron (He-NS) binaries intially evolved using MESA leading to double neutron stars formation. These post-processing scripts calculate the kick velocities imparted onto the newborn neutron stars, kick directions, post-supernova dynamics and the gravitational wave merger timescales.

1. Models.py : In this script we calculate the remnant masses and kick velocities imparted onto the newborn NSs based on the SN and kick prescription from Mandel and Mueller 2020, and calibrations from Kapil et.al. 2021. Post-supernova dynamics is calculated using the analytical formulas from Brand & Podsiadlowski 1994. The GW merger timescale is calculated using a post-Newtonian expression given by Peters 1964

2. Modified_model.py: This script modifies the standard Mandel & Müller (2020) prescription by introducing explicit conditions for systems that undergo ultra-stripped supernovae (USSN) and electron-capture supernovae (ECSN). 
