This repository is meant to hold some modifications of the code L-PICOLA v1.2, December 2015 by 
Howlett. C., Manera. M. & Percival. W. J. (2015), which will be described below.

The directory l-picola_mod was originally copied from the public repository
for the code L-PICOLA at https://github.com/CullanHowlett/l-picola.git.

The source code was modified in order to give the user an option to compute the
peculiar gravitational potential either at particle's positions or at grid nodes
defined through the 'Nmesh' input parameter. 
With this changes the simulation can now (in both snapshot and lightcone mode) additionally compute:
- the gravitational potential on particle's position,
- the gravitational potential on a mesh in cartesian coordinates specified by the parameter 'Nmesh'.

For details about the possible output formats see PAPER. 

For computing the potential at particle's position enable the option POTENTIAL in the Makefile.
This mode can be run in either snapshot or lightcone mode. For the later further enable LIGHTCONE.
The potential of each particle will be written in the same output files along with position and velocity vector.
More details to the output format are presented in the PAPER. 

For computing the potential on a grid defined by the input parameter 'Nmesh'
in the snapshot mode enable the options GONGRID and POTENTIAL in the Makefile.
The simulation will output the gravitational potential on each grid node along with its coordinates, 
thus rendering a scalar field of the potential for each output redshift 
except for the initial redshift as the gravitational potential is not calculated on a grid in the 2LPT method.

For computing the potential on a grid defined by the input parameter 'Nmesh'
in the lightcone mode enable the options LIGHTCONE, G4LIGHTCONE and POTENTIAL in the Makefile.
The simulation will output the gravitational potential on the grid along with its coordinates. 
This routine renders a scalar field of the potential between each pair 
of neighbouring timesteps given by output_redshifts.dat except for the timesteps between 
the initial redshift and the first output redshift. The resulting list of data files provide 
a gravitational potential field structured in redshift shells.
