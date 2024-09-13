import trimesh

import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
import math
import io
import cv2
from PIL import Image
# video stuff
frames = []
reduction_factor = 99

# def get_opencv_img_from_buffer(buffer, flags):
#     bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
#     return cv2.imdecode(bytes_as_np_array, flags)

# def get_opencv_img_from_buffer(og_plt):
#     buf = io.BytesIO()
#     og_plt.savefig(buf, format='png')
#     buf.seek(0)
#     bytes_as_np_array = np.frombuffer(buf.read(), dtype=np.uint8)
#     image = cv2.imdecode(bytes_as_np_array)
#     cv2.imshow("opencv", image)
#     cv2.waitkey(0)
#     return image

def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    # image.show()
    return image




face_color = '#EEE4D8'

# Load the STL or OBJ file
mesh = trimesh.load('./usable_bull.stl')
# mesh = trimesh.load('./bulls/Bull_low_poly_v1.stl')

# Display some basic information about the mesh
print(f'Original mesh has {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.')

amount_to_reduce = mesh.vertices.shape[0] - 11
num_vertices_to_cut = amount_to_reduce / 11

amount_faces_to_reduce = mesh.faces.shape[0] - 11
num_faces_to_cut = amount_faces_to_reduce / 11
print(num_faces_to_cut)

# Simplify the mesh using a fraction of the original number of faces (e.g., 50% reduction)
simplified_mesh = mesh#.simplify_quadric_decimation(face_count = int(mesh.faces.shape[0] * 0.5))

rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.radians(4), direction=[0, 1, 0], point=mesh.centroid)
simplified_mesh = simplified_mesh.apply_transform(rotation_matrix)
# simplified_mesh = simplified_mesh.simplify_quadric_decimation(face_count = 1048)

# Function to plot the mesh with highlighted vertices and edges
def plot_mesh(mesh, ax, show_edges=True, show_vertices=True):
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    ax.set_facecolor(face_color)
    # ax.bo

    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # ax.axes.get_zaxis().set_visible(False)
    
    # Plot the mesh surface
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], cmap='viridis', alpha=0)

    # Optionally, plot the edges
    if show_edges:
        for edge in mesh.edges:
            ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], color='k', linewidth=0.5)
    # ax.set_title(f'{i}_{mesh.faces.shape[0]}')
    

    #Optionally, highlight the vertices
    # if show_vertices:
    #     ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=10)

# Create a grid of subplots
n_rows, n_cols = 3, 4
# fig = plt.figure(figsize=(20, 20))


# 0 1
# 1 5
# 2 9
# 3 2 
# 4 6
# 5 10
# 6 3
# 7 7
# 8 11
# 9 4
# 10 8
# 11 12

# math.floor(i / num_rows) + [col] + 1
# Loop through grid and plot different views of the model
i = 0
for row in range(20):
    for col in range(20):
# for row in range(4):
#     for col in range(3):
        index = row + (col * n_cols) + 1
        # i = i + 1
        # print(int(simplified_mesh.faces.shape[0]),  num_faces_to_cut)
        # print(int(simplified_mesh.faces.shape[0]) - num_faces_to_cut)
        print(int(simplified_mesh.faces.shape[0]))
        # if (int(simplified_mesh.faces.shape[0]) > 0 and index < 12):
        if (int(simplified_mesh.faces.shape[0]) > 3 and index < 100):
            print(f"{i} --- f({col}, {row}) = {index} | {int(simplified_mesh.faces.shape[0])}")


            

            
            old_mesh = simplified_mesh.copy()
            # ax.set_title(old_mesh.faces.shape[0])
            # print(old_mesh.faces.shape[0])

            # # main plot stuff
            # ax = fig.add_subplot(n_rows, n_cols, index, projection='3d')
            # # Plot the rotated mesh with edges and vertices highlighted
            # plot_mesh(old_mesh, ax, show_edges=True, show_vertices=True)

            # # Set the view angle to focus on a specific side of the model
            # # ax.view_init(elev=10, azim=270)
            # ax.axis('off')
            # ax.view_init(elev=-2, azim=270, roll=7.5)
            


            # frame plot stuff
            #fig setup
            fig_2 = plt.figure(figsize=(20, 20))
            sub_index = 1
            for rrow in range(1,4):
                for ccol in range(1,4):
                    if sub_index < 9:
                        #main view
                        ax_2 = fig_2.add_subplot(3, 3, sub_index, projection='3d')
                        ax_2.axis('off')
                        # ax_2.view_init(elev=-2, azim=270 + ((sub_index - 1) * 45), roll=7.5)
                        ax_2.view_init(elev=-2, azim=((270 + ((sub_index - 1) * 45))+(6 * i)), roll=7.5)
                        # ax_2.view_init(elev=-2, azim=270 + ((sub_index - 1) * 45), roll=(abs((4) + sub_index - 9) / 4) * 7.5)
                        plot_mesh(old_mesh, ax_2, show_edges=True, show_vertices=True)
                        sub_index = sub_index + 1


            # #no view init view
            # ax_3 = fig_2.add_subplot(3, 3, 3, projection='3d')
            # ax_3.axis('off')
            # # ax_3.view_init(elev=-2, azim=270, roll=7.5)
            # plot_mesh(old_mesh, ax_3, show_edges=True, show_vertices=True)

            # #front_view
            # ax_4 = fig_2.add_subplot(3, 3, 2, projection='3d')
            # ax_4.axis('off')
            # ax_4.view_init(elev=-2, azim=45)
            # plot_mesh(old_mesh, ax_4, show_edges=True, show_vertices=True)

            fig_2.set_facecolor(face_color)
            frame_image = buffer_plot_and_get(fig_2)
            cv_img = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)
            frames.append(cv_img)
            print(cv_img.shape)

            # fig_2.savefig(f"./individual_outputs/output_{i}.svg")
            plt.close(fig_2)
            # newly_simplified_mesh = old_mesh.simplify_quadric_decimation(face_count = int(old_mesh.faces.shape[0] * 0.6))


            # change plot before going on to next
            newly_simplified_mesh = old_mesh.simplify_quadric_decimation(face_count = int(old_mesh.faces.shape[0] * (reduction_factor / 100)))
            simplified_mesh = newly_simplified_mesh
            i = i+1
            
            #video stuff
            # fig_2.canvas.draw()
            # img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            # frames.append(img_plot)

            # fig_2.close()


# # Display the grid of images
# fig.set_facecolor(face_color)
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# # plt.subplots_adjust(bottom=0, top=0.1, hspace=0)  
# plt.tight_layout()
# plt.savefig("output.png")
# plt.savefig("output.svg")
# # plt.show()

video_dim = (2000,2000)
fps = 25
vidwriter = cv2.VideoWriter(f"output_{fps}_{reduction_factor}_grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim)
print(f"There are {len(frames)} frames")
for frame in frames:
    frame = cv2.resize(frame,(2000,2000))
    vidwriter.write(frame)
vidwriter.release()