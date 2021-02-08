import os
import sys
import math

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Joint:
    def __init__(self, parent, offset, name):
        self.name = name
        self.parent = parent
        self.offset = offset
        self.rotation = None
        self.position = None
        if self.parent == None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

    def calcurate_position(self, euler):
        if self.parent is None:
            self.position = [0, 0, 0]
        else:
            rotM = self.euler2RotationMatrix(euler)
            self.position = self.parent.position + np.dot(rotM, self.offset)

    def euler2RotationMatrix(self, euler):
        x, y, z = euler
        R_x = np.matrix([[1,         0,                  0                   ],
                        [0,         math.cos(x), -math.sin(x) ],
                        [0,         math.sin(x), math.cos(x)  ]
                        ])

        R_y = np.matrix([[math.cos(y),    0,      math.sin(y)  ],
                        [0,                     1,      0                   ],
                        [-math.sin(y),   0,      math.cos(y)  ]
                        ])

        R_z = np.matrix([[math.cos(z),    -math.sin(z),    0],
                        [math.sin(z),    math.cos(z),     0],
                        [0,                     0,                      1]
                        ])

        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R
        

def parse_hierarchy(hierarchy):
    stackOfJoint = []
    dictOfJoint = {}
    orderedName = []
    for i in range(len(hierarchy)):
        line = hierarchy[i]
        if line.find('{') > -1:
            level = line.find('{')
            offset = list(map(float, hierarchy[i+1].strip().split()[1:4]))
            name = hierarchy[i-1].strip().split()[1]
            # Root
            if level == 0:
                dictOfJoint[name] = Joint(parent=None, offset=offset, name=name)
                stackOfJoint.append(name)
            # not Root
            else:
                currentlevel =len(stackOfJoint)
                for i in range(currentlevel - level):
                    stackOfJoint.pop()
                parent = dictOfJoint[stackOfJoint[-1]]
                if hierarchy[i-1].strip().split()[0] == 'End':
                    continue
                    #name = dictOfJoint[parent.name].parent.name + 'EndSite'
                elif name == 'Site':
                    name = parent.name + name
                dictOfJoint[name] = Joint(parent=parent, offset=offset, name=name)
                stackOfJoint.append(name)
            orderedName.append(name)
            i += 3
    return dictOfJoint, orderedName

def main(bvh_path):
    hierarchy = []
    frames = []
    with open(bvh_path, 'r') as f:
        # Get Hierarchy Information
        line = f.readline()
        while(line):
            if line.find('MOTION') > -1:
                break
            hierarchy.append(line)
            line = f.readline()
        frame_nums = int(f.readline().split()[1])
        frame_times = float(f.readline().split()[2])
        # Get Frame Information
        line = f.readline()
        while(line):
            frames.append(list(map(float, line.split())))
            line = f.readline()

    # Make hierarchy tree
    dictOfJoint, orderedName = parse_hierarchy(hierarchy)
    for frame in frames:
        for i in range(len(orderedName)):
            key = orderedName[i] 
            joint = dictOfJoint[key]
            if joint.parent is None:
                joint.calcurate_position([0,0,0])
            else:
                euler_rot = frame[i*3:(i+1)*3]
                joint.calcurate_position(euler_rot)
 
    # Visualize to confirm conversion
    joint_positions = []

    for i in range(len(orderedName)):
       joint = dictOfJoint[orderedName[i]] 
       # Root or junction
       if joint.parent != None 
           x = [joint.position[0,0]]
           y = [joint.position[0,1]]
           z = [joint.potision[0,2]]   
       elif joint.parent != dictOfJoint[orderedName[i-1]]:
           ax.plot(x, y, z, "-o",  ms=4, mew=0.5) 
           x = joints[i:i+1,0]
           y = joints[i:i+1,1]
           z = joints[i:i+1,2]    
       else:
           x = x.append(joint.position[0,0])
           y = x.append(joint.position[0,1])
           z = x.append(joint.potision[0,2])   

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=45, azim=45)
    plt.show()

      
if __name__ == '__main__':
    bvh_path = sys.argv[1]
    main(bvh_path)
