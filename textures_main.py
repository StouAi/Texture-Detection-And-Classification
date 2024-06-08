from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D
from vvrpywork.shapes import Mesh3D, PointSet3D
import numpy as np
import time 
import random
import open3d as o3d
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
WIDTH = 800
HEIGHT = 800

#initializing the scene
class TextureScene(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Textures")
       
        self.smoothed_window = None
        self.reset()

    def reset(self):
        # Generate a list of unique filenames for the mesh files
        def generate_unique_filenames(num_files, max_num):
            filenames = set()
            while len(filenames) < num_files:
                new_file = "plyfiles/"+str(random.randint(1, max_num)) + ".ply"
                filenames.add(new_file)
            return list(filenames)
        
        #initialize the mesh files and future lists
        self.mesh_files = generate_unique_filenames(5, 220)
        self.meshes = []
        self.smoothed_meshes = []
        self.principal_curvatures = []
        self.smooth_principal_curvatures = []
        spacing = 2  # space between the meshes
        for i, mesh_file in enumerate(self.mesh_files):
            
            mesh = Mesh3D(mesh_file)
            mesh.color = [1,0,1,1]
            if i%2 == 0:
                self.move_mesh(mesh, [spacing*i, 0, 0])
            else:
                self.move_mesh(mesh, [-spacing*i, 0,0])
            
            self.meshes.append(mesh)
            self.addShape(mesh, f"mesh_{i}")
        
            #calculate the principal curvatures for each mesh
            self.principal_curvatures.append(self.calculate_principal_curvatures(mesh))
        
        


    def mesh_copy(self,mesh_files):
        mesh_copies = []
        for mesh in mesh_files:
            smoothed_mesh = Mesh3D(mesh)
            smoothed_mesh.color = [1,1,0,1]

            mesh_copies.append(smoothed_mesh)
        
        return mesh_copies

  
    def move_mesh(self,mesh, offset):
        vertices = np.asarray(mesh.vertices)
        vertices += offset
        mesh.vertices = vertices

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.R:
            self.reset()
        elif symbol == Key.ESCAPE:
            self.close()
        elif symbol == Key.L:
            print("Performing Laplacian smoothing")
            self.adjacency_list = []
            for i,mesh in enumerate(self.meshes):
                self.adjacency_list.append(self.find_adjacent_vertices(self.meshes[i])) 
            
    
           
            lambda_factor = 0.5
            iterations = 25
            self.smoothed_meshes = self.mesh_copy(self.mesh_files)
            time_now = time.time()

            for i,mesh in enumerate(self.meshes):
                
                
                
                # Perform Laplacian smoothing
                smoothed_vertices = self.laplacian_smoothing(self.meshes[i], lambda_factor, iterations,self.adjacency_list[i])

                # Update mesh vertices with the smoothed vertices
                self.smoothed_meshes[i].vertices = smoothed_vertices
                self.smooth_principal_curvatures.append(self.calculate_principal_curvatures(self.smoothed_meshes[i]))
            spacing = 2  # Adjust this value as needed
            for i,mesh in enumerate(self.smoothed_meshes):
                if i%2 == 0:
                    self.move_mesh(mesh, [spacing*i, -2, 0])
                else:
                    self.move_mesh(mesh, [-spacing*i, -2,0])
                self.addShape(mesh,f"smoothed mesh_{i}")
            print("Time taken: ", time.time()-time_now)
        
        elif symbol == Key.S:
            print("Calculating saliency")
            k = 5 #nearest neighbors
            knns = []
            time_now = time.time()
            for i,mesh in enumerate(self.meshes):

                knns = self.knn_search(mesh,k)
                normals = np.asarray(mesh.vertex_normals)
                saliency_vals = self.saliency_python(normals, knns)
                saliency_vals_normalized = (saliency_vals - np.min(saliency_vals)) / (np.max(saliency_vals) - np.min(saliency_vals))
                
                self.visualize_metric(mesh,saliency_vals_normalized)
            
                self.updateShape(f"mesh_{i}")
                
            print("Time taken: ", time.time()-time_now)

       
            if len(self.smoothed_meshes) > 0:
                print("visualizing smoothed meshes")
                for i,smooth_mesh in enumerate(self.smoothed_meshes):
                    knns = self.knn_search(smooth_mesh,k)
                    normals = np.asarray(smooth_mesh.vertex_normals)
                    saliency_vals = self.saliency_python(normals, knns)
                    saliency_vals_normalized = (saliency_vals - np.min(saliency_vals)) / (np.max(saliency_vals) - np.min(saliency_vals))
                    self.visualize_metric(smooth_mesh,saliency_vals_normalized)
                    self.updateShape(f"smoothed mesh_{i}")
                
            print("Time taken: ", time.time()-time_now)
            
                
        elif symbol == Key.M:
            print("Calculating mean curvature")
            time_now = time.time()
            
            
            for i, mesh in enumerate(self.meshes):
                mean_curvature_values = self.mean_curvature(mesh,i)
                mean_curvature_normalized = (mean_curvature_values - np.min(mean_curvature_values)) / (np.max(mean_curvature_values) - np.min(mean_curvature_values))
                self.visualize_metric(mesh,mean_curvature_normalized)
                self.updateShape(f"mesh_{i}")
                
            
            print("visualizing smoothed meshes")
            if len(self.smoothed_meshes) > 0:
                for i, smooth_mesh in enumerate(self.smoothed_meshes):
                    mean_curvature_values = self.mean_curvature(smooth_mesh,i,True)
                    mean_curvature_normalized = (mean_curvature_values - np.min(mean_curvature_values)) / (np.max(mean_curvature_values) - np.min(mean_curvature_values))
                    self.visualize_metric(smooth_mesh,mean_curvature_normalized)
                    self.updateShape(f"smoothed mesh_{i}")
                    

            print("Time taken: ", time.time()-time_now)
        elif symbol == Key.G:
            print("Calculating Gaussian curvature")
            time_now = time.time()
            # Compute Gaussian curvature
            for i,mesh in enumerate(self.meshes):
                gaussian_curvature_values = self.gaussian_curvature(mesh,i)
                gaussian_curvature_normalized = (gaussian_curvature_values - np.min(gaussian_curvature_values)) / (np.max(gaussian_curvature_values) - np.min(gaussian_curvature_values))
                self.visualize_metric(mesh,gaussian_curvature_normalized)
                self.updateShape(f"mesh_{i}")
                
           

            if len(self.smoothed_meshes) > 0:
                print("visualizing smoothed meshes")
                for i, smooth_mesh in enumerate(self.smoothed_meshes):
                    gaussian_curvature_values = self.gaussian_curvature(smooth_mesh,i,True)
                    gaussian_curvature_normalized = (gaussian_curvature_values - np.min(gaussian_curvature_values)) / (np.max(gaussian_curvature_values) - np.min(gaussian_curvature_values))
                    self.visualize_metric(smooth_mesh,gaussian_curvature_normalized)
                    self.updateShape(f"smoothed mesh_{i}")
                    print(gaussian_curvature_normalized[0:10])
               
            print("Time taken: ", time.time()-time_now)
                    
    


    
    def calculate_principal_curvatures(self, mesh: Mesh3D) -> np.ndarray:
        """Calculate principal curvatures at each vertex of the mesh."""
        vertices = np.asarray(mesh.vertices)
        adjacency_list = self.find_adjacent_vertices(mesh)
        num_vertices = len(vertices)
        principal_curvatures = np.zeros((num_vertices, 2))  # Assuming 2 principal curvatures per vertex

        # Precompute covariance matrices for all neighboring vertices
        covariance_matrices = np.array([np.cov(vertices[neighbors], rowvar=False) for neighbors in adjacency_list])

        for i, neighbors in enumerate(adjacency_list):
            # Compute eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices[i])

            # Sort eigenvalues and eigenvectors in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Store principal curvatures
            principal_curvatures[i] = eigenvalues[:2]  # Assuming 2 principal curvatures

        return principal_curvatures



    def gaussian_curvature(self, mesh,mesh_index,smooth=False):
        """Calculate Gaussian curvature at each vertex of the mesh."""
        vertices = np.asarray(mesh.vertices)
        adjacency_list = self.find_adjacent_vertices(mesh)
        num_vertices = len(vertices)
        gaussian_curvature = np.zeros(num_vertices)

        # Compute principal curvatures
        if smooth:
            principal_curvatures = self.smooth_principal_curvatures[mesh_index]

        else:
            principal_curvatures = self.principal_curvatures[mesh_index]
        
        
        for i, neighbors in enumerate(adjacency_list):
            # Use the product of principal curvatures to compute Gaussian curvature
            gaussian_curvature[i] = principal_curvatures[i, 0] * principal_curvatures[i, 1]

       
        return gaussian_curvature


            

    def mean_curvature(self, mesh,mesh_index,smooth=False):
        vertices = np.asarray(mesh.vertices)
        num_vertices = len(vertices)
        adjacency_list = self.find_adjacent_vertices(mesh)

        # Compute vertex normals if not already computed
        mean_curvature = np.zeros(num_vertices)

        if smooth:
            
            principal_curvatures = self.smooth_principal_curvatures[mesh_index]

        else:
            principal_curvatures = self.principal_curvatures[mesh_index]

        for i, neighbors in enumerate(adjacency_list):
            # Compute mean curvature using the principal curvatures
            mean_curvature[i] = 0.5 * np.sum(principal_curvatures[i])

        return mean_curvature

    
        
            
    def knn_search(self,mesh,k):
        knn_tree = o3d.geometry.KDTreeFlann(mesh._shape)
        knns = []
        vertices = np.asarray(mesh.vertices)

        for v in vertices:
            [kt,ids,_] = knn_tree.search_knn_vector_3d(v,k)
            knns.append(ids)

        knns = np.stack(knns)

        return knns


    def saliency_python(self,normals, knns):
    
        N = normals[knns]

        saliency_vals = np.zeros((normals.shape[0]))

        # parse each vertex of the mesh and use the normals of the neighbor to compute the covariance matrix
        for i, n in enumerate(N):
                
            
            cov = np.dot(n.transpose(), n)
        
            eigenvalues, _ = np.linalg.eig(cov)
        
            eig1, eig2, eig3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
            saliency = 1 / sqrt(eig1 * eig1 + eig2 * eig2 + eig3 * eig3)
        
            saliency_vals[i] = saliency
        
        saliency_norm = (saliency_vals - saliency_vals.min()) / (saliency_vals.max() - saliency_vals.min())

        return saliency_norm
    
    def visualize_metric(self, mesh,metric_normal):
        
            mesh.color = [1,1,1,1]
            
            vertex_colors = np.asarray(mesh.vertex_colors)  
            colormap = plt.get_cmap('viridis')  # Get the colormap
            colors = colormap(metric_normal)[:,:3]
            for idx,color in enumerate(colors):
                vertex_colors[idx] = color
            
            mesh.vertex_colors = vertex_colors.tolist()
           
            
        

    def find_adjacent_vertices(self, mesh):
        num_vertices = len(mesh.vertices)
        adjacency_list = [[] for _ in range(num_vertices)]

        # Convert triangles to a NumPy array for efficient indexing
        triangles = np.array(mesh.triangles)

        # Create the adjacency list
        for triangle in triangles:
            i, j, k = triangle
            adjacency_list[i].extend([j, k])
            adjacency_list[j].extend([i, k])
            adjacency_list[k].extend([i, j])

        # Convert the lists to sets to remove duplicates, then convert back to lists
        return [list(set(neighbors)) for neighbors in adjacency_list]

    def laplacian_smoothing(self, mesh, lambda_factor, iterations,adjacency_list):
        vertices = np.asarray(mesh.vertices)
        num_vertices = len(vertices)
        # Convert adjacency list to a format  for indexing
        adjacency_indices = np.zeros((num_vertices, max(map(len, adjacency_list))), dtype=int)
        for i, neighbors in enumerate(adjacency_list):
            adjacency_indices[i, :len(neighbors)] = neighbors

        # Create a mask to avoid invalid indexing 
        mask = adjacency_indices != 0

        for _ in range(iterations):
            # Calculate the centroids using  indexing and masking
            neighbor_vertices = vertices[adjacency_indices]
            centroids = np.sum(neighbor_vertices * mask[:, :, None], axis=1) / mask.sum(axis=1)[:, None]

            # Perform the smoothing operation
            vertices += lambda_factor * (centroids - vertices)
            
        return vertices
if __name__ == "__main__":
    app = TextureScene()
    app.mainLoop()
