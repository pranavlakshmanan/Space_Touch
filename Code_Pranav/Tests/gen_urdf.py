#!/usr/bin/env python3
import xacro
# process the XACRO file
doc = xacro.process_file("allegro_hand_description_left_digit.xacro")

# output the expanded URDF
print(doc.toxml())
