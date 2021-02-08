import sys, os, glob

import numpy as np

#['GlobalRoothipPos  Root hips  LHipJoint LeftUpLeg LeftLeg LeftFoot LeftToeBase EndSite RHipJoint']

class Normalize():
    def parse_initial_pos(self,bvh_path):
        offsets = []
        with open(bvh_path, 'r') as f:
            line = f.readline()
            while line:
                if 'OFFSET' in line:
                    pos = line.strip().split(' ')[1:]
                    pos = list(map(float, pos))
                    offsets.append(pos)
                line = f.readline()
        offsets = np.array(offsets)
        return offsets


    def readtxt(self,text_path):
        with open(text_path, 'r') as f:
            positions = []
            line = f.readline()
            line = f.readline()
            while line:
                pos = line.strip().split('  ')
                pos = [list(map(float, p.split(' '))) for p in pos]
                if len(positions) > 1:
                    if length != len(pos):
                        break
                length = len(pos)
                positions.append(np.array(pos))
                line = f.readline()
                if len(line) > 1000:
                    break
            positions = np.array(positions)
        return positions

    def convert_to_relative(self,global_positions):
        positions = np.copy(global_positions)

        #from Root(1) to RightHandEndSite(9)
        for i in range(0,8):
            positions[:,9-i,:] = positions[:,9-i,:] - positions[:,9-(i+1),:]

        #from Thoat(3) to LeftHandEndSite(15) 
        for i in range(5):
            positions[:,15-i,:] = positions[:,15-i,:] - positions[:,15-(i+1),:]
        positions[:,10,:] = positions[:,10,:] - positions[:,3,:]

        #from Thoat(3) to HeadEndSite(18)
        for i in range(3):
            positions[:,18-i,:] = positions[:,18-i,:] - positions[:,18-(i+1),:]
        positions[:,16,:] = positions[:,16,:] - positions[:,3,:]

        #from Root(1) to RightToesEndSite(24)
        for i in range(5):
            positions[:,24-i,:] = positions[:,24-i,:] - positions[:,24-(i+1),:]
        positions[:,19,:] = positions[:,19,:] - positions[:,1,:]

        #from Root(1) to LeftToesEndSite(30)
        for i in range(5):
            positions[:,30-i,:] = positions[:,30-i,:] - positions[:,30-(i+1),:]
        positions[:,25,:] = positions[:,25,:] - positions[:,1,:]
 
        # set root to 0
        positions[:,1,:] = np.zeros((positions.shape[0],3))
        return positions

    def normalize_by_offset(self,positions, offsets):
        norm_positions = np.zeros(positions.shape)
        norm_positions[:,0,:] = positions[:,0,:]
        for i in range(1, positions.shape[1]):
            #offsetのnormが0のものを省く
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                offset_norm = np.linalg.norm(offsets[i-1,:])
                norm_positions[:,i,:] = positions[:,i,:] / offset_norm
            else:
                norm_positions[:,i,:] = np.zeros((positions.shape[0],3))
        #print(f'pre norm {positions[1,3,:]}, offset {offsets[2]}, after norm {norm_positions[1,3,:]}')
        return norm_positions

    def denormalization_by_offset(self,norm_positions, offsets):
        positions = np.zeros(norm_positions.shape)
        positions[:,0,:] = norm_positions[:,0,:]
        for i in range(1, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                offset_norm = np.linalg.norm(offsets[i-1,:])
                for j in range(positions.shape[0]):
                    norm = np.linalg.norm(norm_positions[j,i,:])
                    if norm > 1e-6:
                        positions[j,i,:] = (norm_positions[j,i,:] / norm) * offset_norm
                    else:
                        positions[j,i,:] = np.zeros(3)
            else:
                positions[:,i,:] = np.zeros((norm_positions.shape[0], 3))
        #print(f'pre denorm {norm_positions[1,3,:]}, offset {offsets[2]}, after denorm {positions[1,3,:]}')
        return positions

    def convert_to_global(self,relative_positions, root):
        positions = np.copy(relative_positions)

        positions[:,1,:] = root 
        #from Root(1) to RightHandEndSite(10)
        for i in range(1,10):
            positions[:,i+1,:] = positions[:,i+1,:] + positions[:,i,:]

        #from Thoat(3) to LeftHandEndSite(15)
        positions[:,10,:] = positions[:,10,:] + positions[:,3,:]
        for i in range(5):
            positions[:,10+(i+1),:] = positions[:,10+(i+1),:] + positions[:,10+i,:]

        #from Thoat(3) to HeadEndSite(18)
        positions[:,16,:] = positions[:,16,:] + positions[:,3,:]
        for i in range(3):
            positions[:,16+(i+1),:] = positions[:,16+(i+1),:] + positions[:,16+i,:]

        #from Root(1) to RightToesEndSite(24)
        positions[:,19,:] = positions[:,19,:] + positions[:,1,:]
        for i in range(5):
            positions[:,19+(i+1),:] = positions[:,19+(i+1),:] + positions[:,19+i,:]

        #from Root(1) to LeftToesEndSite(30)
        positions[:,25,:] = positions[:,25,:] + positions[:,1,:]
        for i in range(5):
            positions[:,25+(i+1),:] = positions[:,25+(i+1),:] + positions[:,25+i,:]

        print(positions[0,0:20,:])
        return positions

    def zero_cut(self,positions, offsets):
        size = 0
        for j in range(offsets.shape[0]):
            if np.sum(np.sum(np.abs(offsets[j,:]), axis=0), axis=0) > 0:
                size += 1
        non_zero_positions = np.zeros((positions.shape[0], size+1, 3))
        non_zero_positions[:,0,:] = positions[:,0,:]
        count = 1
        for i in range(1, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                non_zero_positions[:,count,:] = positions[:,i,:]
                count += 1
        return non_zero_positions

    def zero_add(self,non_zero_positions, offsets):
        positions = np.zeros((non_zero_positions.shape[0], offsets.shape[0]+1, 3))
        positions[:,0,:] = non_zero_positions[:,0,:]
        positions[:,1,:] = np.zeros((non_zero_positions.shape[0],3))
        count = 1
        for i in range(2, offsets.shape[0]+1):
            if np.sum(np.sum(np.abs(offsets[i-1,:]), axis=0), axis=0) > 0:
                positions[:,i,:] = non_zero_positions[:,count,:]
                count += 1
            else:
                positions[:,i,:] = positions[:,i-1,:]
        return positions

def main():
    textlist = sorted(glob.glob('../../data/bvh/style-dataset_jp/bvh/*/*.txt'))

    n = Normalize()

    for textpath in textlist:
        print(textpath)
        positions = n.readtxt(textpath)
        bvh_path = os.path.splitext(textpath)[0] + '.bvh'
        offsets = n.parse_initial_pos(bvh_path)
        relative_positions = n.convert_to_relative(positions)
        #norm_positions = n.normalize_by_offset(relative_positions, offsets)
        #root_norm_positions = n.convert_to_global(norm_positions, np.zeros((norm_positions.shape[0],3)))  #腰をルートとして、全関節の長さを１に正規化した座標系
        #non_zero_root_norm_positions = n.zero_cut(root_norm_positions, offsets)
        ##np.save(os.path.splitext(textpath)[0] + 'n.npy', non_zero_root_norm_positions)
        #root_norm_positions = n.zero_add(non_zero_root_norm_positions, offsets)
        #rec_norm_positions = n.convert_to_relative(root_norm_positions)
        #denorm_relative_positions = n.denormalization_by_offset(rec_norm_positions, offsets)
        #denorm_relative_positions = n.denormalization_by_offset(norm_positions, offsets)
        #global_positions = n.convert_to_global(denorm_relative_positions, positions[:,0,:])
        global_positions = n.convert_to_global(relative_positions, positions[:,1,:])
        name = os.path.split(os.path.splitext(textpath)[0])[1]
        print(global_positions.shape, positions[0,20,:],global_positions[0,20,:])
        sys.exit()
        top, _ = os.path.split(textpath)
        _, stylename = os.path.split(top)

        if not os.path.exists(f'data/train_jp/style-dataset_jp/bvh/{stylename}'):
            os.makedirs(f'data/train_jp/style-dataset_jp/bvh/{stylename}')
        if not os.path.exists(f'data/test_jp/style-dataset_jp/bvh/{stylename}'):
            os.makedirs(f'data/test_jp/style-dataset_jp/bvh/{stylename}')

#    npy_list = sorted(glob.glob('data/bvh/style-dataset_jp/bvh/*/*.npy'))
#    for npy_path in npy_list:
#        npy = np.load(npy_path)
#        top, name = os.path.split(npy_path)
#        top, motion_name = os.path.split(top)
#
#        if not os.path.exists(f'data/train_jp/CMU_jp/Styled_jp/Subject137'):
#            os.makedirs(f'data/train_jp/CMU_jp/Styled_jp/Subject137')
#        if not os.path.exists(f'data/test_jp/CMU_jp/Styled_jp/Subject137'):
#            os.makedirs(f'data/test_jp/CMU_jp/Styled_jp/Subject137')
#
#        length = npy.shape[0]
#        train = npy[:int(length*0.9),:,:]
#        np.save(f'data/train_jp/CMU_jp/Styled_jp/Subject137/{name}', train)
#        test = npy[int(length*0.9):,:,:]
#       np.save(f'data/test_jp/CMU_jp/Styled_jp/Subject137/{name}', npy)


if __name__ == '__main__':
    main()
