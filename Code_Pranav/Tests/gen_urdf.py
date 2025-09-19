#!/usr/bin/env python3
import xacro
# process the XACRO file
doc = xacro.process_file("Code_Pranav/Test_Structure_files/6_DOF_Base_joint.xacro")

# output the expanded URDF
print(doc.toxml())
