import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN, KMeans

from filter import (
    resolve_path,
    remove_background,
    filter_noise,
)


load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


# ============================================================
# K PARA CADA ESCENA
# ============================================================

K_VALUES = {
    "1_coche.csv": 1,
    "coche_coche.csv": 2,
    "coche_coche_moto.csv": 3,
    "pointcloud_1686138334837467970.csv": 1,
}


# ============================================================
# PARÁMETROS DE LIMPIEZA CON DBSCAN
# ============================================================

NOISE_EPS = 0.35
NOISE_MIN_SAMPLES = 8


# ============================================================
# CREAR POINT CLOUD
# ============================================================

def create_pointcloud(
    data: pd.DataFrame,
    color: list[float]
) -> o3d.geometry.PointCloud:

    pcd = o3d.geometry.PointCloud()

    points = data[
        ["x", "y", "z"]
    ].to_numpy()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    colors = np.tile(
        color,
        (len(points), 1)
    )

    pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

    return pcd


# ============================================================
# QUITAR RUIDO CON DBSCAN
# ============================================================

def remove_dbscan_noise(
    data: pd.DataFrame,
    eps: float = NOISE_EPS,
    min_samples: int = NOISE_MIN_SAMPLES
) -> pd.DataFrame:

    if len(data) == 0:
        return data

    points = data[
        ["x", "y", "z"]
    ].to_numpy()

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(
        points
    )

    result = data.copy()

    result["noise_cluster"] = labels

    total = len(result)

    noise = len(
        result[
            result["noise_cluster"] == -1
        ]
    )

    clean = result[
        result["noise_cluster"] != -1
    ].copy()

    clean.drop(
        columns=["noise_cluster"],
        inplace=True
    )

    print()
    print("Filtrado DBSCAN:")
    print(f"  eps = {eps}")
    print(f"  min_samples = {min_samples}")
    print(f"  Puntos antes: {total}")
    print(f"  Ruido eliminado: {noise}")
    print(f"  Puntos restantes: {len(clean)}")

    return clean


# ============================================================
# K-MEANS
# ============================================================

def apply_kmeans(
    data: pd.DataFrame,
    k: int
) -> pd.DataFrame:

    if len(data) == 0:
        return data

    points = data[
        ["x", "y", "z"]
    ].to_numpy()

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        points
    )

    result = data.copy()

    result["cluster"] = labels

    return result


def apply_car_car_motorcycle(data: pd.DataFrame) -> pd.DataFrame:
    """Separa los coches y un candidato pequeño alejado de ambos.

    Los 22 puntos conocidos orientan el tamaño, pero no identifican
    por sí solos la moto: un fragmento del coche también puede tener 22.
    """
    labels = DBSCAN(eps=0.75, min_samples=3).fit_predict(
        data[["x", "y", "z"]].to_numpy()
    )
    grouped = data.assign(group=labels)
    sizes = grouped.groupby("group").size().drop(-1, errors="ignore")
    car_groups = sizes[sizes > 100].nlargest(2).index
    if len(car_groups) != 2:
        raise ValueError("No se han encontrado los dos coches de esta escena.")
    centers = grouped.groupby("group")[["x", "y", "z"]].mean()
    candidates = []
    for group_id in sizes[(sizes >= 16) & (sizes <= 33)].index:
        distance = np.linalg.norm(
            centers.loc[car_groups].to_numpy() - centers.loc[group_id].to_numpy(),
            axis=1,
        ).min()
        if distance >= 10.0:
            candidates.append(group_id)
    if len(candidates) != 1:
        raise ValueError("No hay un único candidato lejano compatible con la moto.")
    motorcycle_id = candidates[0]
    cars = apply_kmeans(data.loc[np.isin(labels, car_groups)].copy(), k=2)
    motorcycle = data.loc[labels == motorcycle_id].copy()
    motorcycle["cluster"] = 2
    print(f"Candidato a moto lejano: {len(motorcycle)} puntos (referencia: 22).")
    return pd.concat([cars, motorcycle]).sort_index()


# ============================================================
# MOSTRAR CLUSTERS
# ============================================================

def show_clusters(
    data: pd.DataFrame,
    title: str,
    k: int
) -> None:

    colors = [
        [1.0, 0.0, 0.0],     # rojo
        [0.0, 1.0, 0.0],     # verde
        [0.0, 0.0, 1.0],     # azul
        [1.0, 0.5, 0.0],     # naranja
        [1.0, 0.0, 1.0],     # magenta
        [0.0, 1.0, 1.0],     # cyan
    ]

    geometries = []

    print()
    print(
        f"Vehículos detectados con K-Means: {k}"
    )

    # ========================================================
    # CADA CLUSTER = UN VEHÍCULO
    # ========================================================

    for cluster_id in range(k):

        cluster = data[
            data["cluster"] == cluster_id
        ].copy()

        if len(cluster) == 0:
            continue

        color = colors[
            cluster_id % len(colors)
        ]

        pcd = create_pointcloud(
            cluster,
            color
        )

        geometries.append(
            pcd
        )

        center = cluster[
            ["x", "y", "z"]
        ].mean().to_numpy()

        print()
        print(
            ("Moto" if title == "coche_coche_moto.csv" and cluster_id == 2
             else f"Vehículo {cluster_id + 1}")
        )

        print(
            f"  Cluster K-Means: {cluster_id}"
        )

        print(
            f"  Puntos: {len(cluster)}"
        )

        print(
            "  Centro aproximado:"
        )

        print(
            f"    X = {center[0]:.2f} m"
        )

        print(
            f"    Y = {center[1]:.2f} m"
        )

        print(
            f"    Z = {center[2]:.2f} m"
        )

    # ========================================================
    # EJES
    # ========================================================

    axes = (
        o3d.geometry
        .TriangleMesh
        .create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0]
        )
    )

    geometries.append(
        axes
    )

    # ========================================================
    # MOSTRAR OPEN3D
    # ========================================================

    if len(geometries) > 1:

        o3d.visualization.draw_geometries(
            geometries,
            window_name=(
                f"K-Means - {title}"
            ),
            width=1200,
            height=800
        )

    else:

        print(
            "No hay clusters para mostrar."
        )


# ============================================================
# PROCESAR ARCHIVO
# ============================================================

def process_file(
    path: Path
) -> None:

    print()
    print("=" * 65)

    print(
        f"K-MEANS: {path.name}"
    )

    print("=" * 65)

    if path.name == "carretera.csv":

        print(
            "carretera.csv no contiene vehículos."
        )

        return

    if path.name not in K_VALUES:

        raise ValueError(
            f"No existe un valor de K configurado "
            f"para '{path.name}'"
        )

    k = K_VALUES[
        path.name
    ]

    print(
        f"Número de clusters configurado: K = {k}"
    )

    # --------------------------------------------------------
    # Cargar frame
    # --------------------------------------------------------

    data = pd.read_csv(
        path
    )

    print(
        f"Puntos originales: {len(data)}"
    )

    # --------------------------------------------------------
    # Cargar fondo
    # --------------------------------------------------------

    background_path = (
        DATA_DIR / "carretera.csv"
    )

    background = pd.read_csv(
        background_path
    )

    # --------------------------------------------------------
    # Quitar fondo
    # --------------------------------------------------------

    if path.name == "coche_coche_moto.csv":
        # Un fondo sin retorno (range=0) no demuestra que el punto sea fondo.
        # Conservamos esos candidatos y eliminamos los grupos residuales después.
        if len(data) != len(background):
            raise ValueError("El frame y el fondo deben tener igual número de filas.")
        current = data["range"].to_numpy(dtype=float)
        reference = background["range"].to_numpy(dtype=float)
        keep = (
            np.isfinite(current) & np.isfinite(reference) & (current > 0)
            & ((reference == 0) | ((reference > 0) & (reference - current > 300)))
        )
        data = data.loc[keep].copy()
    else:
        data = remove_background(data=data, background=background, threshold=300)

    print(
        f"Puntos después de quitar fondo: "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # Filtro básico
    # --------------------------------------------------------

    data = filter_noise(
        data
    )

    print(
        f"Puntos después del filtrado básico: "
        f"{len(data)}"
    )

    if len(data) == 0:

        print(
            "No quedan puntos después del filtrado."
        )

        return

    # --------------------------------------------------------
    # DBSCAN PARA QUITAR RUIDO
    # --------------------------------------------------------

    if path.name == "coche_coche_moto.csv":
        data = apply_car_car_motorcycle(data)
    else:
        data = remove_dbscan_noise(
            data=data,
            eps=NOISE_EPS,
            min_samples=NOISE_MIN_SAMPLES
        )

        if len(data) == 0:

            print(
                "DBSCAN ha eliminado todos los puntos."
            )

            return

        # --------------------------------------------------------
        # K-MEANS
        # --------------------------------------------------------

        data = apply_kmeans(
            data=data,
            k=k
        )

    # --------------------------------------------------------
    # Información
    # --------------------------------------------------------

    print()
    print(
        "Distribución K-Means:"
    )

    for cluster_id in range(k):

        count = len(
            data[
                data["cluster"] == cluster_id
            ]
        )

        print(
            f"  Cluster {cluster_id}: "
            f"{count} puntos"
        )

    # --------------------------------------------------------
    # Mostrar
    # --------------------------------------------------------

    show_clusters(
        data=data,
        title=path.name,
        k=k
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    filename = os.getenv(
        "POINTCLOUD_FILE"
    )

    if not filename:

        raise ValueError(
            "No está definida POINTCLOUD_FILE "
            "en el fichero .env"
        )

    path = resolve_path(
        filename
    )

    process_file(
        path
    )


if __name__ == "__main__":
    main()
