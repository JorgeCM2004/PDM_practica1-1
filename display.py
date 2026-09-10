import os
from pathlib import Path

import matplotlib as mpl
import open3d as o3d
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(__file__).parent / "data"


def resolve_path(name: str) -> Path:
    for candidate in (Path(name), DATA_DIR / name, DATA_DIR / f"{name}.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No existe '{name}' en {DATA_DIR}")


def colorize(data: pd.DataFrame, field: str) -> pd.DataFrame:
    values = data[field]
    normalized = (values - values.min()) / (values.max() - values.min())
    return mpl.colormaps["viridis"](normalized)[:, :3]


def show_pointcloud(data: pd.DataFrame, title: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[["x", "y", "z"]].to_numpy())
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main() -> None:
    filename = os.getenv("POINTCLOUD_FILE")

    path = resolve_path(filename)
    data = pd.read_csv(path)
    show_pointcloud(data, title=path.name)


if __name__ == "__main__":
    main()
