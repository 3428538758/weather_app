# fire_damage_pred/data_preprocessing.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from utils import (
    read_fire_points, read_weather, read_buildings,
    build_fire_building_map, build_time_series_features
)

def main():
    # ---------- 0. 参数 ----------
    buffer_m = 500        # 初始缓冲半径（米）
    grid_size = 100.0     # 网格分辨率（米/像素）
    h = w = 10            # 网格尺寸
    # ---------- 1. 读取数据 ----------
    fire_gdf = read_fire_points("C:/Users/34285/Desktop/新建文件夹 (4)/fire_palisades.csv")
    weather_df = read_weather("C:/Users/34285/Desktop/新建文件夹 (4)/palisades_weather.csv")
    bld_gdf   = read_buildings("C:/Users/34285/Desktop/新建文件夹 (4)/2023_Buildings_with_DINS_data.geojson")
    # ---------- 2. 火灾—建筑关联 ----------
    building_fire_events = build_fire_building_map(bld_gdf, fire_gdf, buffer_m)
    # ---------- 3. 构建时序特征 ----------
    X_list, y_list = [], []
    for idx, fires_df in building_fire_events.items():
        result = build_time_series_features(
            idx, fires_df, bld_gdf, weather_df,
            grid_size, h, w
        )
        if result:
            X_list.append(result[0])
            y_list.append(result[1])
    # ---------- 4. 统一时间长度 & 保存 ----------

    # 4.1 先统计所有建筑序列里最大的时间步数 T_max
    T_max = max(feat.shape[0] for feat in X_list)

    def pad_sequence(arr, T_target):
        """把 (T, C, H, W) 张量在时间维度末尾用 0 填到同一长度"""
        T_curr = arr.shape[0]
        if T_curr == T_target:
            return arr
        pad_shape = (T_target - T_curr, *arr.shape[1:])
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    # 4.2 对每个建筑做 padding，再 stack 成一个大数组
    X_padded = np.stack([pad_sequence(feat, T_max) for feat in X_list])  # (N, T_max, C, H, W)
    y_arr = np.array(y_list, dtype=np.int64)

    # 4.3 保存
    np.save("features.npy", X_padded)
    np.save("labels.npy", y_arr)
    print("Saved features.npy & labels.npy")
    print("Final tensor shape:", X_padded.shape, "Labels shape:", y_arr.shape)



if __name__ == "__main__":
    main()
