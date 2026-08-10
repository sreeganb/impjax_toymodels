# Complete skeleton for the implementation of the code in plain english text

## IMP is used to read in input information such as pdb files, topology files,      experimental data such as crosslinking data, cryo em densities, APMS data and so on.
## User will use the run script to do this part, generation of the IMP input representation as well as definition of the scoring function and generate the JAX restraint scoring function which is JIT compiled so that external samplers can come in and simply use this object along with the degrees of freedom and run sampling using modern samplers. 
## The functions have to be very small and do specific tasks, for example, there has to be a function in the wrapper that simply reads in a model, hierarchy and extract all sorts of information about the system such as rigid body indices, rigid body coordinates, flexible beads coordinates and indices, and unit tests are added to properly make sure that this part is well tested.
## There has to be function that will list out all of the particles (rigid bodies as well as flexible beads that are to be sampled.)
 
