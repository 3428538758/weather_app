# fire_damage_pred/utils.py
"""
公共工具函数：
1. 读取火灾点 CSV  -> GeoDataFrame (UTM 投影)
2. 读取天气 CSV   -> 按小时插值后的 DataFrame（index = Timestamp）
3. 读取建筑 GeoJSON -> GeoDataFrame (UTM 投影，含 label)
4. 建立建筑-火灾映射  -> {建筑 idx: 火灾点 GeoDataFrame}
5. 构建建筑时序特征   -> (T, C, H, W) numpy + label
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, Tuple, Optional

# ------------------ 常量 ------------------ #
damage_mapping: Dict[str, int] = {
    "No Damage": 0,
    "Affected": 1,
    "Minor (10-25%)": 2,
    "Major (26-50%)": 3,
    "Destroyed (>50%)": 4,
    "Inaccessible": 5
}

WEATHER_RENAME = {
    "temperature_2m (°C)": "temp",
    "relative_humidity_2m (%)": "humidity",
    "precipitation (mm)": "precip",
    "rain (mm)": "rain",
    "wind_speed_10m (km/h)": "wind_speed",
    "wind_direction_10m (°)": "wind_dir"
}

WEATHER_FEATURES = ["temp", "humidity", "precip", "wind_speed", "wind_dir"]

# ------------------ 读取火灾点 ------------------ #
def read_fire_points(
    csv_path: str,
    time_col: str = "datetime",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    intensity_col: str = "frp",
    time_fmt: str = "%Y/%m/%d %H:%M",
    dst_epsg: int = 32611
) -> gpd.GeoDataFrame:
    """读取火灾点 CSV 为 GeoDataFrame，并投影到指定坐标系。"""
    df = pd.read_csv(csv_path)

    # 1) 解析时间字符串
    df["time"] = pd.to_datetime(df[time_col], format=time_fmt, errors="coerce")

    # 2) 清洗经纬度：强制转 float，无法转换的设 NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # 3) 丢弃缺失值行
    df = df.dropna(subset=["time", lat_col, lon_col]).reset_index(drop=True)

    # 4) 没有强度列时填默认值 1.0
    if intensity_col not in df.columns:
        df[intensity_col] = 1.0

    # 5) GeoDataFrame + 投影
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    ).to_crs(dst_epsg)

    gdf = gdf.rename(columns={intensity_col: "intensity"})
    return gdf[["time", "intensity", "geometry"]]

# ------------------ 读取天气 ------------------ #
def read_weather(
    csv_path: str,
    time_col: str = "time",
    rename_dict: Dict[str, str] = WEATHER_RENAME,
    freq: str = "1h"
) -> pd.DataFrame:
    """读取天气 CSV，重采样为 1 小时分辨率并线性插值。"""
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns=rename_dict)
    df = df.set_index(time_col)
    df = df[WEATHER_FEATURES]            # 保留需要的列
    df = df.resample(freq).interpolate(method="time")
    return df

# ------------------ 读取建筑 ------------------ #
def read_buildings(
    geojson_path: str,
    damage_col: str = "DAMAGE",
    dst_epsg: int = 32611
) -> gpd.GeoDataFrame:
    """读取建筑 GeoJSON，生成中心点并添加数值标签。"""
    gdf = gpd.read_file(geojson_path).set_crs(4326).to_crs(dst_epsg)
    gdf["center"] = gdf.geometry.centroid
    gdf["label"] = gdf[damage_col].map(damage_mapping).astype("Int64")
    gdf = gdf.dropna(subset=["label"]).reset_index(drop=True)
    return gdf

# ------------------ 建立建筑-火灾映射 ------------------ #
def build_fire_building_map(
    bld_gdf: gpd.GeoDataFrame,
    fire_gdf: gpd.GeoDataFrame,
    buffer_m: float
) -> Dict[int, gpd.GeoDataFrame]:
    """返回 {建筑 idx: 其缓冲区内火灾点 GeoDataFrame}。"""
    mapping: Dict[int, gpd.GeoDataFrame] = {}
    sindex = fire_gdf.sindex
    for idx, bld in bld_gdf.iterrows():
        buf_poly = bld.geometry.buffer(buffer_m)
        poss = list(sindex.intersection(buf_poly.bounds))
        fires = fire_gdf.iloc[poss]
        fires = fires[fires.geometry.within(buf_poly)].sort_values("time")
        mapping[idx] = fires
    return mapping

# ------------------ 构建时序特征 ------------------ #
def build_time_series_features(
    building_idx: int,
    fires_df: gpd.GeoDataFrame,
    bld_gdf: gpd.GeoDataFrame,
    weather_df: pd.DataFrame,
    grid_m: float,
    H: int,
    W: int,
    time_margin_h: int = 1
) -> Optional[Tuple[np.ndarray, int]]:
    """返回 (features, label)。无火灾时返回 None。"""
    if len(fires_df) == 0:
        return None

    # 1) 生成时间序列
    t_start = fires_df["time"].iloc[0] - pd.Timedelta(hours=time_margin_h)
    t_end   = fires_df["time"].iloc[-1] + pd.Timedelta(hours=time_margin_h)
    timestamps = pd.date_range(t_start, t_end, freq="1h")
    T = len(timestamps)
    C = 1 + len(WEATHER_FEATURES)

    # 2) 初始化特征张量
    feats = np.zeros((T, C, H, W), dtype=np.float32)

    # 3) 建筑中心
    center: Point = bld_gdf.loc[building_idx, "center"]
    cx, cy = center.x, center.y
    half_w = W / 2 * grid_m
    half_h = H / 2 * grid_m

    # 4) 填充时间步
    win = pd.Timedelta(minutes=30)
    for t_i, ts in enumerate(timestamps):
        # 4.1 火灾通道
        active = fires_df[(fires_df["time"] >= ts - win) & (fires_df["time"] <= ts + win)]
        for _, fire in active.iterrows():
            dx = fire.geometry.x - cx
            dy = fire.geometry.y - cy
            if -half_w <= dx <= half_w and -half_h <= dy <= half_h:
                ix = int((dx + half_w) // grid_m)
                iy = int((dy + half_h) // grid_m)
                if 0 <= ix < W and 0 <= iy < H:
                    feats[t_i, 0, iy, ix] += fire["intensity"]

                # utils.py 片段 —— 找到 for t_i, ts in enumerate(timestamps): 的循环，
                # 把其中 “4.2 天气通道” 这一小段替换为 ↓

                # 4.2 天气通道
                if ts in weather_df.index:
                    row = weather_df.loc[ts]
                else:
                    nearest_idx = weather_df.index.get_indexer([ts], method="nearest")[0]
                    row = weather_df.iloc[nearest_idx]

                # 如果 row 是 DataFrame(1 行)，压缩成 Series
                if isinstance(row, pd.DataFrame):
                    row = row.squeeze()

                for ch, key in enumerate(WEATHER_FEATURES, start=1):
                    feats[t_i, ch, :, :] = float(row[key])

    label = int(bld_gdf.loc[building_idx, "label"])
    return feats, label
