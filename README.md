# Mounting Algorithm

Algorithm assists user into finding mounting locations of strain gauges without returning to the FEA of the structure. The Nastran Input File (either DAT or BDF file) is required as well as a list of CQUAD4 element IDs and its respective orientations with respect to its elemental coordinate system. The algorithm will write a .txt file with the information on where to mount a strain gauge and at what orientation in terms of a chosen global coordinate system.
