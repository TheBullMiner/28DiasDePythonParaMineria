import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def create_anisotropic_covariance(points, C0, C1, a_major, a_minor, angle):
    """Crea una matriz de covarianza anisotrópica."""
    angle_rad = np.deg2rad(angle)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])
    
    dist_matrix = squareform(pdist(points[:, :2]))
    
    # Rotar y escalar coordenadas para anisotropía
    coords_rot = np.dot(points[:,:2], rot_matrix)
    h_x = squareform(pdist(coords_rot[:, 0][:, None]))
    h_y = squareform(pdist(coords_rot[:, 1][:, None]))
    
    # Calcular la distancia anisotrópica
    h_aniso = np.sqrt((h_x / a_minor)**2 + (h_y / a_major)**2)
    
    # Modelo esférico
    covariance = np.where(
        h_aniso <= 1,
        C0 + C1 * (1.5 * (h_aniso) - 0.5 * (h_aniso)**3),
        C0 + C1
    )
    np.fill_diagonal(covariance, C0 + C1)
    return covariance

# Parámetros
n_points = 500
C0, C1 = 0, 1 # Nugget y Sill
a_major, a_minor = 60, 20 # Rangos de la anisotropía
angle = 45 # Dirección de máxima continuidad

# Generar puntos
np.random.seed(42)
points = np.random.rand(n_points, 2) * 200

# Crear la matriz de covarianza y simular
cov_matrix = create_anisotropic_covariance(points, C0, C1, a_major, a_minor, angle)
sim_values = np.random.multivariate_normal(np.zeros(n_points), cov_matrix)

# Crear DataFrame y guardar
df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'value': sim_values})
df.to_csv('data/sample_data.csv', index=False)
print("Datos anisotrópicos simulados creados en 'data/sample_data.csv'")