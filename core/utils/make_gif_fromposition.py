# import sys, os, glob

# import cv2
# import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import subprocess
# import shutil

# def show_pose(positions, path, delay=2, dirpath='tmp', nogrid=False, save_frames=False, lw=3, elev=45, azim=45):
#     frames = []
#     dirpath_base = dirpath
#     i = 0
#     dirpath = os.path.join(dirpath_base, str(i))
#     while os.path.exists(dirpath):
#         dirpath = os.path.join(dirpath_base, str(i))
#         i += 1
#     os.makedirs(dirpath)

#     rangex = np.max(np.max(positions[:,:,0],axis=0),axis=0) - np.min(np.min(positions[:,:,0],axis=0),axis=0)
#     rangey = np.max(np.max(positions[:,:,2],axis=0),axis=0) - np.min(np.min(positions[:,:,2],axis=0),axis=0)
#     rangez = np.max(np.max(positions[:,:,1],axis=0),axis=0) - np.min(np.min(positions[:,:,1],axis=0),axis=0)
#     rangemax = min(max(rangex, rangey, rangez),30)
#     avex = np.average(np.average(positions[:,:,0], axis=0), axis=0)
#     avey = np.average(np.average(positions[:,:,2], axis=0), axis=0)
#     avez = np.average(np.average(positions[:,:,1], axis=0), axis=0)


#     for i in range(positions.shape[0]):
#         fig = plt.figure()
#         ax = Axes3D(fig)
#         ax.view_init(elev=elev, azim=azim)
#         if nogrid:
#             ax.grid(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             for key, val in ax.spines.items():
#                 ax.spines[key].set_color("none")
#             ax.w_xaxis.set_pane_color((0.,0.,0.,0.))
#             ax.w_yaxis.set_pane_color((0.,0.,0.,0.))
#             ax.w_zaxis.set_pane_color((0.,0.,0.,0.))
#         ax.set_xlim(avex-rangemax//2, avex+rangemax//2)
#         ax.set_ylim(avey-rangemax//2, avey+rangemax//2)
#         ax.set_zlim(avez-rangemax//2, avez+rangemax//2)
#         pose = positions[i,:,:]

#         x = pose[30:39,0]
#         y = pose[30:39,2]
#         z = pose[30:39,1]
#         ax.plot(x,y,z,"-o",color='black',ms=4,mew=0.5,linewidth=lw)
#         x = pose[8:14,0]
#         y = pose[8:14,2]
#         z = pose[8:14,1]
#         ax.plot(x,y,z,"-o",color='blue',ms=4,mew=0.5,linewidth=lw)
#         x = pose[14:21,0]
#         y = pose[14:21,2]
#         z = pose[14:21,1]
#         ax.plot(x,y,z,"-o",color='green',ms=4,mew=0.5,linewidth=lw)
#         x = pose[:8,0]
#         y = pose[:8,2]
#         z = pose[:8,1]
#         ax.plot(x,y,z,"-o",color='red',ms=4,mew=0.5,linewidth=lw)
#         x = pose[21:30,0]
#         y = pose[21:30,2]
#         z = pose[21:30,1]
#         ax.plot(x,y,z,"-o",color='orange',ms=4,mew=0.5,linewidth=lw)

#         plt.savefig(f'{dirpath}/{i:05d}.png')
#         if save_frames:
#             plt.savefig(f'{os.path.splitext(path)[0]}_{i:02d}.png')
#         plt.close()

#     cmd = ['convert','-layers','optimize','-loop','0','-delay', f'{delay}',f'{dirpath}/*.png',f'{path}']
#     subprocess.run(cmd)
#     shutil.rmtree(dirpath)


# def show_pose_multi(positions_list, path, delay=2, dirpath='tmp', save_frames=False, nogrid=False, lw=6, elev=15, azim=45, layered=False):
#     frames = []
#     dirpath_base = dirpath
#     i = 0
#     dirpath = os.path.join(dirpath_base, str(i))
#     while os.path.exists(dirpath):
#         dirpath = os.path.join(dirpath_base, str(i))
#         i += 1
#     os.makedirs(dirpath)

#     positions_list_array = np.array(positions_list)
#     rangex = np.max(np.max(np.max(positions_list_array[:,:,:,0], axis=0),axis=0),axis=0) - np.min(np.min(np.min(positions_list_array[:,:,:,0], axis=0),axis=0),axis=0)
#     rangey = np.max(np.max(np.max(positions_list_array[:,:,:,2], axis=0),axis=0),axis=0) - np.min(np.min(np.min(positions_list_array[:,:,:,2], axis=0),axis=0),axis=0)
#     rangez = np.max(np.max(np.max(positions_list_array[:,:,:,1], axis=0),axis=0),axis=0) - np.min(np.min(np.min(positions_list_array[:,:,:,1], axis=0),axis=0),axis=0)
#     rangemax = min(max(rangex, rangey, rangez),30)
#     avex = np.average(np.average(np.average(positions_list_array[:,:,:,0], axis=0), axis=0), axis=0)
#     avey = np.average(np.average(np.average(positions_list_array[:,:,:,2], axis=0), axis=0), axis=0)
#     avez = np.average(np.average(np.average(positions_list_array[:,:,:,1], axis=0), axis=0), axis=0)

#     frame_length = positions_list[0].shape[0]

#     for j in range(len(positions_list)):
#         positions = positions_list[j]
#         for i in range(frame_length):
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.view_init(elev=elev, azim=azim)
#             if nogrid:
#                 ax.grid(False)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax.set_zticks([])
#                 for key, val in ax.spines.items():
#                     ax.spines[key].set_color("none")
#                 ax.w_xaxis.set_pane_color((0.,0.,0.,0.))
#                 ax.w_yaxis.set_pane_color((0.,0.,0.,0.))
#                 ax.w_zaxis.set_pane_color((0.,0.,0.,0.))
#             ax.set_xlim(avex-rangemax//2, avex+rangemax//2)
#             ax.set_ylim(avey-rangemax//2, avey+rangemax//2)
#             ax.set_zlim(avez-rangemax//2, avez+rangemax//2)
#             pose = positions[i,:,:]
#             x = pose[30:39,0]
#             y = pose[30:39,2]
#             z = pose[30:39,1]
#             ax.plot(x,y,z,"-o",color='black',ms=4,mew=0.5,linewidth=lw)
#             x = pose[8:14,0]
#             y = pose[8:14,2]
#             z = pose[8:14,1]
#             ax.plot(x,y,z,"-o",color='blue',ms=4,mew=0.5,linewidth=lw)
#             x = pose[14:21,0]
#             y = pose[14:21,2]
#             z = pose[14:21,1]
#             ax.plot(x,y,z,"-o",color='green',ms=4,mew=0.5,linewidth=lw)
#             x = pose[:8,0]
#             y = pose[:8,2]
#             z = pose[:8,1]
#             ax.plot(x,y,z,"-o",color='red',ms=4,mew=0.5,linewidth=lw)
#             x = pose[21:30,0]
#             y = pose[21:30,2]
#             z = pose[21:30,1]
#             ax.plot(x,y,z,"-o",color='orange',ms=4,mew=0.5,linewidth=lw)
#             plt.savefig(f'{dirpath}/{i:05d}_{j:02d}.png')

#             plt.clf()
#             plt.cla()
#             ax.clear()
#             plt.close()
	
#     if layered:
#         for i in range(positions_list[0].shape[0]):
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.view_init(elev=elev, azim=azim)
#             if nogrid:
#                 ax.grid(False)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax.set_zticks([])
#                 for key, val in ax.spines.items():
#                     ax.spines[key].set_color("none")
#                 ax.w_xaxis.set_pane_color((0.,0.,0.,0.))
#                 ax.w_yaxis.set_pane_color((0.,0.,0.,0.))
#                 ax.w_zaxis.set_pane_color((0.,0.,0.,0.))
#             ax.set_xlim(avex-rangemax//2, avex+rangemax//2)
#             ax.set_ylim(avey-rangemax//2, avey+rangemax//2)
#             ax.set_zlim(avez-rangemax//2, avez+rangemax//2)
#             for j in range(len(positions_list)):
#                 pose = positions_list[j][i,:,:]
#                 x = pose[30:39,0]
#                 y = pose[30:39,2]
#                 z = pose[30:39,1]
#                 ax.plot(x,y,z,"-o",color='black',ms=4,mew=0.5,linewidth=lw)
#                 x = pose[8:14,0]
#                 y = pose[8:14,2]
#                 z = pose[8:14,1]
#                 ax.plot(x,y,z,"-o",color='blue',ms=4,mew=0.5,linewidth=lw)
#                 x = pose[14:21,0]
#                 y = pose[14:21,2]
#                 z = pose[14:21,1]
#                 ax.plot(x,y,z,"-o",color='green',ms=4,mew=0.5,linewidth=lw)
#                 x = pose[:8,0]
#                 y = pose[:8,2]
#                 z = pose[:8,1]
#                 ax.plot(x,y,z,"-o",color='red',ms=4,mew=0.5,linewidth=lw)
#                 x = pose[21:30,0]
#                 y = pose[21:30,2]
#                 z = pose[21:30,1]
#             ax.plot(x,y,z,"-o",color='orange',ms=4,mew=0.5,linewidth=lw)
#             plt.savefig(f'{dirpath}/{i:05d}_layered.png')
#             plt.close()


#     if not os.path.exists(dirpath):
#         os.makedirs(dirpath)

#     for i in range(frame_length):
#         for j in range(len(positions_list)):
#             if j == 0:
#                 im = cv2.imread(f'{dirpath}/{i:05d}_{j:02d}.png')
#             else:
#                 im = cv2.hconcat([im, cv2.imread(f'{dirpath}/{i:05d}_{j:02d}.png')])
#         cv2.imwrite(f'{dirpath}/concat_{i:05d}.png', im)

#     if layered:
#         layered_path = os.path.splitext(path)[0] + '_layered.gif'
#         cmd = ['convert','-layers','optimize','-loop','0','-delay', f'{delay}',f'{dirpath}/*_layered.png',f'{layered_path}']
#         subprocess.run(cmd)

#     cmd = ['convert','-layers','optimize','-loop','0','-delay', f'{delay}',f'{dirpath}/concat_*.png',f'{path}']
#     subprocess.run(cmd)
#     shutil.rmtree(dirpath)

# def show_pose_morphing(positions_list, path, delay=2, dirpath='tmp', nogrid=False, lw=6, elev=45, azim=45):
#     frames = []
#     dirpath_base = dirpath
#     i = 0
#     while os.path.exists(dirpath):
#         dirpath = os.path.join(dirpath_base, str(i))
#         i += 1
#     os.makedirs(dirpath)

#     rangex = np.max(np.max(positions_list[0][:,:,0],axis=0),axis=0) - np.min(np.min(positions_list[0][:,:,0],axis=0),axis=0)
#     rangey = np.max(np.max(positions_list[0][:,:,2],axis=0),axis=0) - np.min(np.min(positions_list[0][:,:,2],axis=0),axis=0)
#     rangez = np.max(np.max(positions_list[0][:,:,1],axis=0),axis=0) - np.min(np.min(positions_list[0][:,:,1],axis=0),axis=0)
#     rangemax = min(max(rangex, rangey, rangez),30)
#     avex = np.average(np.average(positions_list[0][:,:,0], axis=0), axis=0)
#     avey = np.average(np.average(positions_list[0][:,:,2], axis=0), axis=0)
#     avez = np.average(np.average(positions_list[0][:,:,1], axis=0), axis=0)

#     for i in range(positions_list[0].shape[0]):
#         fig = plt.figure()
#         ax = Axes3D(fig)
#         ax.view_init(elev=elev, azim=azim)
#         if nogrid:
#             ax.grid(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             for key, val in ax.spines.items():
#                 ax.spines[key].set_color("none")
#             ax.w_xaxis.set_pane_color((0.,0.,0.,0.))
#             ax.w_yaxis.set_pane_color((0.,0.,0.,0.))
#             ax.w_zaxis.set_pane_color((0.,0.,0.,0.))
#         ax.set_xlim(avex-rangemax//2, avex+rangemax//2)
#         ax.set_ylim(avey-rangemax//2, avey+rangemax//2)
#         ax.set_zlim(avez-rangemax//2, avez+rangemax//2)
#         for j in range(len(positions_list)):
#             pose = positions_list[j][i,:,:]
#             x = pose[30:39,0]
#             y = pose[30:39,2]
#             z = pose[30:39,1]
#             ax.plot(x,y,z,"-o",color='black',ms=4,mew=0.5,linewidth=lw)
#             x = pose[8:14,0]
#             y = pose[8:14,2]
#             z = pose[8:14,1]
#             ax.plot(x,y,z,"-o",color='blue',ms=4,mew=0.5,linewidth=lw)
#             x = pose[14:21,0]
#             y = pose[14:21,2]
#             z = pose[14:21,1]
#             ax.plot(x,y,z,"-o",color='green',ms=4,mew=0.5,linewidth=lw)
#             x = pose[:8,0]
#             y = pose[:8,2]
#             z = pose[:8,1]
#             ax.plot(x,y,z,"-o",color='red',ms=4,mew=0.5,linewidth=lw)
#             x = pose[21:30,0]
#             y = pose[21:30,2]
#             z = pose[21:30,1]
#             ax.plot(x,y,z,"-o",color='orange',ms=4,mew=0.5,linewidth=lw)
#         plt.savefig(f'{dirpath}/{i:05d}.png')
#         plt.close()

#     cmd = ['convert','-layers','optimize','-loop','0','-delay', f'{delay}',f'{dirpath}/*.png',f'{path}']
#     subprocess.run(cmd)
#     shutil.rmtree(dirpath)

# if __name__ == '__main__':
#     npy_path_raw = sys.argv[1]
#     npy_path_list = npy_path_raw.split(',')
#     depth = 3
#     positions = []
#     for npy_path in npy_path_list:
#         position = np.load(npy_path)
#         positions.append(position)
#     head,filename = os.path.split(npy_path_list[0])
#     filename, ext = os.path.splitext(filename)
#     path = head + '/' + filename[:-2] + 'overlay.gif'
#     delay = 64 // (2 ** (depth + 1))
#     show_poses_over(positions, path, delay=delay, dirpath='testtmp')
