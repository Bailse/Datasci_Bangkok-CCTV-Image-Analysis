import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from data_loader import load_enriched_data, load_supplementary_data

st.set_page_config(
    page_title="Bangkok CCTV Traffic Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 14px 18px;
        border-radius: 8px;
    }
    .insight-box {
        background-color: rgba(59, 130, 246, 0.06);
        border-left: 3px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 1rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df, data_source_msg = load_enriched_data()
extras = load_supplementary_data()

if "hour" not in df.columns:
    if "hour_local" in df.columns:
        df["hour"] = pd.to_numeric(df["hour_local"], errors="coerce").fillna(0).astype(int)
    else:
        df["hour"] = 0

if "camera_label" not in df.columns:
    df["camera_label"] = df["location"] if "location" in df.columns else df["camera_id"]

metric_candidates = [c for c in ("vehicles_total", "car", "motor") if c in df.columns]
if not metric_candidates:
    st.error("No vehicle count columns (vehicles_total, car, motor) found in the dataset.")
    st.stop()

st.title("Bangkok CCTV Traffic Analytics")
st.caption(f"2110403 DSDE-CEDT Project  |  {data_source_msg}")

st.sidebar.header("Filters")

metric_labels = {
    "vehicles_total": "Total Vehicles",
    "car": "Cars",
    "motor": "Motorcycles",
}
selected_metric = st.sidebar.selectbox(
    "Traffic Metric",
    options=metric_candidates,
    format_func=lambda x: metric_labels.get(x, x),
    index=0,
)

preset = st.sidebar.radio(
    "Time Window",
    options=[
        "08:00 - 10:00 (Morning)",
        "14:00 - 16:00 (Afternoon)",
        "17:00 - 20:00 (Evening)",
        "00:00 - 23:59 (All Day)",
        "Custom",
    ],
    index=3,
)

preset_ranges = {
    "08:00 - 10:00 (Morning)": (8, 10),
    "14:00 - 16:00 (Afternoon)": (14, 16),
    "17:00 - 20:00 (Evening)": (17, 20),
    "00:00 - 23:59 (All Day)": (0, 23),
}

if preset in preset_ranges:
    start_hr, end_hr = preset_ranges[preset]
else:
    start_hr, end_hr = st.sidebar.slider("Hour Range (0 - 23)", 0, 23, (8, 10))

if start_hr > end_hr:
    start_hr, end_hr = end_hr, start_hr

base_df = df.copy()
base_df["camera_id"] = base_df["camera_id"].astype(str)
base_df["metric_val"] = pd.to_numeric(base_df[selected_metric], errors="coerce").fillna(0.0)

chosen_day = "All"
if "day_type" in base_df.columns:
    day_types = ["All"] + sorted(base_df["day_type"].dropna().astype(str).unique().tolist())
    chosen_day = st.sidebar.selectbox("Day Type", options=day_types, index=0)
    if chosen_day != "All":
        base_df = base_df[base_df["day_type"].astype(str) == chosen_day]

chosen_rain = "All"
if "rain_group" in base_df.columns:
    rain_groups = ["All"] + sorted(base_df["rain_group"].dropna().astype(str).unique().tolist())
    chosen_rain = st.sidebar.selectbox("Weather / Rain Group", options=rain_groups, index=0)
    if chosen_rain != "All":
        base_df = base_df[base_df["rain_group"].astype(str) == chosen_rain]

chosen_road = "All"
if "road_class_300m" in base_df.columns:
    road_classes = ["All"] + sorted(base_df["road_class_300m"].dropna().astype(str).unique().tolist())
    chosen_road = st.sidebar.selectbox("Road Class (OSM 300m)", options=road_classes, index=0)
    if chosen_road != "All":
        base_df = base_df[base_df["road_class_300m"].astype(str) == chosen_road]

window_df = base_df[(base_df["hour"] >= start_hr) & (base_df["hour"] <= end_hr)].copy()
if window_df.empty:
    st.warning("No records match the selected filters and time window.")
    st.stop()

agg_spec = {
    "avg_traffic": ("metric_val", "mean"),
    "max_traffic": ("metric_val", "max"),
    "records": ("metric_val", "count"),
}
if "car" in window_df.columns:
    agg_spec["car_avg"] = ("car", "mean")
if "motor" in window_df.columns:
    agg_spec["motor_avg"] = ("motor", "mean")
if "road_class_300m" in window_df.columns:
    agg_spec["road_class"] = ("road_class_300m", "first")
if "osm_ways_300m" in window_df.columns:
    agg_spec["osm_ways_300m"] = ("osm_ways_300m", "first")
if "mot_reports_300m_hour" in window_df.columns:
    agg_spec["accident_reports"] = ("mot_reports_300m_hour", "sum")

cam_summary = (
    window_df.groupby(["camera_id", "camera_label"], as_index=False)
    .agg(**agg_spec)
    .sort_values("avg_traffic", ascending=False)
    .reset_index(drop=True)
)

if {"latitude", "longitude"}.issubset(base_df.columns):
    coords = (
        base_df[["camera_id", "latitude", "longitude"]]
        .assign(
            latitude=lambda d: pd.to_numeric(d["latitude"], errors="coerce"),
            longitude=lambda d: pd.to_numeric(d["longitude"], errors="coerce"),
        )
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("camera_id")
    )
    cam_summary = cam_summary.merge(coords, on="camera_id", how="left")


def traffic_points(vals):
    lo = math.floor(float(vals.min()))
    hi = math.ceil(float(vals.max()))
    if lo >= hi:
        return [lo, lo + 1]
    step = max(1, math.ceil((hi - lo) / 20))
    pts = list(range(lo, hi + 1, step))
    if pts[-1] != hi:
        pts.append(hi)
    return sorted(set(pts))


stops = traffic_points(cam_summary["avg_traffic"])
traffic_range = st.sidebar.select_slider(
    "Mean Traffic Range",
    options=stops,
    value=(stops[0], stops[-1]),
)

filtered_cams = cam_summary[
    (cam_summary["avg_traffic"] >= traffic_range[0])
    & (cam_summary["avg_traffic"] <= traffic_range[1])
].copy()

if filtered_cams.empty:
    st.warning("No cameras match the selected traffic range.")
    st.stop()

MAP_STYLES = {
    "Dark": pdk.map_styles.CARTO_DARK,
    "Light": pdk.map_styles.CARTO_LIGHT,
    "Road": pdk.map_styles.CARTO_ROAD,
}
map_style = st.sidebar.selectbox("Base Map Style", options=list(MAP_STYLES), index=0)

with st.sidebar.expander("Spatial Model Parameters", expanded=False):
    st.markdown("**DBSCAN Clustering**")
    eps_degrees = st.slider("eps (degree)", 0.002, 0.030, 0.012, 0.002, format="%.3f")
    min_samples = st.slider("min_samples", 2, 15, 3, 1)
    num_top_clusters = st.slider("Top Clusters to Show", 1, 15, 5, 1)
    st.divider()
    st.markdown("**Kernel Density Estimation (KDE)**")
    bandwidth = st.slider("bandwidth", 0.002, 0.040, 0.012, 0.002, format="%.3f")

top_cam = filtered_cams.iloc[0]
hourly_all = base_df.groupby("hour", as_index=False)["metric_val"].mean().sort_values("hour")
peak_row = hourly_all.loc[hourly_all["metric_val"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Cameras", f"{len(filtered_cams)}", f"{int(filtered_cams['records'].sum()):,} records")
c2.metric("Mean Traffic", f"{filtered_cams['avg_traffic'].mean():.2f}", f"{start_hr:02d}:00 - {end_hr:02d}:59")
c3.metric("Peak Hour (All Day)", f"{int(peak_row['hour']):02d}:00", f"Avg {peak_row['metric_val']:.2f}")
c4.metric("Top Hotspot", f"{top_cam['camera_id']}", f"Avg {top_cam['avg_traffic']:.2f}")

tab_overview, tab_spatial, tab_kde, tab_external, tab_model = st.tabs(
    [
        "Overview & Insights",
        "Spatial Hotspots (DBSCAN & Accidents)",
        "Density Analysis (KDE)",
        "External Factors (5 Sources)",
        "Model & Pipeline Evaluation",
    ]
)

with tab_overview:
    car_share = 0.0
    if {"car", "motor"}.issubset(window_df.columns):
        total_cm = window_df["car"].sum() + window_df["motor"].sum()
        if total_cm > 0:
            car_share = (window_df["car"].sum() / total_cm) * 100

    day_night_note = ""
    if "day_night" in base_df.columns and base_df["day_night"].nunique() > 1:
        dn_means = base_df.groupby("day_night")["metric_val"].mean()
        if "day" in dn_means and "night" in dn_means:
            day_night_note = (
                f"- <b>Day vs Night Volume:</b> Daytime average is <b>{dn_means['day']:.2f}</b> "
                f"vs nighttime average <b>{dn_means['night']:.2f}</b> vehicles/frame.<br/>"
            )

    st.markdown(
        f"""
        <div class="insight-box">
            <b>Analytical Summary ({start_hr:02d}:00 - {end_hr:02d}:59 | Day: {chosen_day} | Weather: {chosen_rain} | Road: {chosen_road})</b><br/>
            - <b>Highest Traffic Location:</b> Camera <b>{top_cam['camera_id']}</b> ({top_cam['camera_label']}) with a mean of <b>{top_cam['avg_traffic']:.2f}</b> vehicles/frame (peak: {top_cam['max_traffic']:.0f}).<br/>
            - <b>Peak Hour:</b> Highest average volume occurs at <b>{int(peak_row['hour']):02d}:00</b> (mean: {peak_row['metric_val']:.2f} vehicles/frame).<br/>
            - <b>Vehicle Composition:</b> Cars account for <b>{car_share:.1f}%</b> and motorcycles for <b>{100 - car_share:.1f}%</b> in this window.<br/>
            {day_night_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig_hourly = px.line(
            hourly_all,
            x="hour",
            y="metric_val",
            markers=True,
            title=f"Hourly Traffic Pattern ({start_hr:02d}:00 - {end_hr:02d}:59 highlighted)",
            labels={"hour": "Hour of Day (0 - 23)", "metric_val": "Mean Count / Image"},
        )
        fig_hourly.add_vrect(x0=start_hr, x1=end_hr, fillcolor="orange", opacity=0.18, line_width=0)
        fig_hourly.update_layout(xaxis=dict(dtick=2), height=400)
        st.plotly_chart(fig_hourly, width="stretch")

    with col_right:
        if {"car", "motor"}.issubset(window_df.columns):
            comp_df = pd.DataFrame(
                {
                    "Vehicle Type": ["Cars", "Motorcycles"],
                    "Mean Count": [window_df["car"].mean(), window_df["motor"].mean()],
                }
            )
            fig_pie = px.pie(
                comp_df,
                names="Vehicle Type",
                values="Mean Count",
                hole=0.45,
                title="Vehicle Composition in Selected Window",
                color_discrete_sequence=["#3b82f6", "#f59e0b"],
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, width="stretch")

    top_n_chart = min(15, len(filtered_cams))
    top_chart_df = filtered_cams.head(top_n_chart).copy()
    fig_bar = px.bar(
        top_chart_df,
        x="camera_id",
        y="avg_traffic",
        color="avg_traffic",
        color_continuous_scale="OrRd",
        hover_data=["camera_label", "max_traffic", "records"],
        title=f"Top {top_n_chart} Traffic Hotspots ({start_hr:02d}:00 - {end_hr:02d}:59)",
        labels={"camera_id": "Camera ID", "avg_traffic": "Mean Count"},
    )
    fig_bar.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig_bar, width="stretch")

valid_geo = filtered_cams.dropna(subset=["latitude", "longitude"]).copy()

with tab_spatial:
    if valid_geo.empty:
        st.warning("No valid latitude/longitude coordinates available for map layers.")
    else:
        view_lat = float(valid_geo["latitude"].mean())
        view_lon = float(valid_geo["longitude"].mean())

        ctrl_col1, ctrl_col2 = st.columns([3, 1.5])
        with ctrl_col1:
            map_mode = st.radio(
                "Spatial Visualization Layer",
                ["DBSCAN Hotspot Clusters", "Traffic Intensity Heatmap", "3D Hexagon Density"],
                horizontal=True,
            )
        with ctrl_col2:
            show_accidents = st.checkbox(
                "Overlay 2026 Bangkok Accidents",
                value=False,
                disabled="accidents" not in extras,
            )

        accident_layer = None
        if show_accidents and "accidents" in extras:
            acc_geo = extras["accidents"].dropna(subset=["latitude", "longitude"]).copy()
            accident_layer = pdk.Layer(
                "ScatterplotLayer",
                acc_geo,
                get_position=["longitude", "latitude"],
                get_color=[239, 68, 68, 130],
                get_radius=65,
                pickable=False,
            )

        if map_mode == "DBSCAN Hotspot Clusters":
            try:
                coords = valid_geo[["latitude", "longitude"]].values
                db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)
                valid_geo["cluster"] = db.labels_

                clusters_count = valid_geo["cluster"].value_counts()
                clusters_count = clusters_count[clusters_count.index != -1]
                top_clusters = clusters_count.head(num_top_clusters)

                unique_clusters = valid_geo[valid_geo["cluster"].isin(top_clusters.index)]["cluster"].unique()
                colormap = plt.get_cmap("hsv")
                cluster_colors = {
                    int(cluster): [int(x * 255) for x in colormap(i / max(1, len(unique_clusters)))[:3]] + [200]
                    for i, cluster in enumerate(unique_clusters)
                }

                viz_data = valid_geo.copy()
                viz_data["color"] = viz_data["cluster"].map(
                    lambda c: cluster_colors.get(int(c), [148, 163, 184, 150])
                )
                viz_data["radius_scaled"] = (viz_data["avg_traffic"].clip(lower=1.0) * 18).clip(110, 700)

                cluster_layer = pdk.Layer(
                    "ScatterplotLayer",
                    viz_data,
                    get_position=["longitude", "latitude"],
                    get_color="color",
                    get_radius="radius_scaled",
                    pickable=True,
                )
                layers = [accident_layer, cluster_layer] if accident_layer else [cluster_layer]

                st.pydeck_chart(
                    pdk.Deck(
                        layers=layers,
                        initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11, pitch=0),
                        map_style=MAP_STYLES[map_style],
                        tooltip={
                            "html": "<b>Cluster:</b> {cluster}<br/>"
                                    "<b>Camera:</b> {camera_id}<br/>"
                                    "<b>Mean Traffic:</b> {avg_traffic}<br/>"
                                    "<b>Location:</b> {camera_label}"
                        },
                    ),
                    height=560,
                )

                if top_clusters.empty:
                    st.info("No dense clusters formed under current filters/eps; showing unclustered cameras in gray.")
                else:
                    st.caption("Colored circles indicate top DBSCAN spatial clusters; gray circles indicate unclustered cameras (-1).")
                    with st.expander("View DBSCAN Cluster Statistics", expanded=False):
                        for cluster_id in top_clusters.index:
                            cluster_data = viz_data[viz_data["cluster"] == cluster_id]
                            st.markdown(f"**Cluster {cluster_id}** ({len(cluster_data)} cameras)")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Mean Traffic", f"{cluster_data['avg_traffic'].mean():.2f}")
                            m2.metric("Peak Traffic", f"{cluster_data['max_traffic'].max():.0f}")
                            m3.metric("Unique Locations", cluster_data["camera_label"].nunique())
                            st.dataframe(
                                cluster_data[["camera_id", "camera_label", "avg_traffic", "max_traffic", "records"]].head(5),
                                width="stretch",
                            )
                            st.divider()
            except Exception as e:
                st.error(f"Error in clustering analysis: {e}")

        elif map_mode == "Traffic Intensity Heatmap":
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                valid_geo,
                get_position=["longitude", "latitude"],
                get_weight="avg_traffic",
                opacity=0.8,
                pickable=True,
            )
            layers = [heatmap_layer, accident_layer] if accident_layer else [heatmap_layer]
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11, pitch=0),
                    map_style=MAP_STYLES[map_style],
                ),
                height=560,
            )

        else:
            hexagon_layer = pdk.Layer(
                "HexagonLayer",
                valid_geo,
                get_position=["longitude", "latitude"],
                radius=600,
                elevation_scale=8,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                opacity=0.65,
            )
            layers = [accident_layer, hexagon_layer] if accident_layer else [hexagon_layer]
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11, pitch=35),
                    map_style=MAP_STYLES[map_style],
                    tooltip={"html": "<b>Cameras in Hex:</b> {elevationValue}"},
                ),
                height=560,
            )

        if show_accidents and "accidents" in extras:
            st.caption(f"Overlay active: {len(extras['accidents']):,} MOT 2026 Bangkok accident locations shown as small red markers.")

with tab_kde:
    if valid_geo.empty:
        st.warning("No valid coordinates for KDE analysis.")
    else:
        try:
            view_lat = float(valid_geo["latitude"].mean())
            view_lon = float(valid_geo["longitude"].mean())
            coords = valid_geo[["latitude", "longitude"]].values

            kde = KernelDensity(bandwidth=bandwidth).fit(coords)
            density = np.exp(kde.score_samples(coords))
            valid_geo["density"] = density
            denom = density.max() - density.min()
            valid_geo["density_normalized"] = (density - density.min()) / denom if denom > 0 else 0.5
            valid_geo["density_formatted"] = valid_geo["density"].apply(lambda x: f"{x:.4f}")

            def density_to_color(val):
                return [int(val * 255), int((1 - abs(2 * val - 1)) * 255), int((1 - val) * 255), 180]

            valid_geo["density_color"] = valid_geo["density_normalized"].apply(density_to_color)

            kde_layer = pdk.Layer(
                "ScatterplotLayer",
                valid_geo,
                get_position=["longitude", "latitude"],
                get_color="density_color",
                get_radius=180,
                pickable=True,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[kde_layer],
                    initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11, pitch=0),
                    map_style=MAP_STYLES[map_style],
                    tooltip={
                        "html": "<b>Camera:</b> {camera_id}<br/>"
                                "<b>Location:</b> {camera_label}<br/>"
                                "<b>Mean Traffic:</b> {avg_traffic}<br/>"
                                "<b>Density:</b> {density_formatted}"
                    },
                ),
                height=520,
            )
            st.caption("Color gradient indicates spatial camera density: blue (low) -> green (medium) -> red (high).")

            k1, k2, k3 = st.columns(3)
            k1.metric("Min Density", f"{valid_geo['density'].min():.4f}")
            k2.metric("Mean Density", f"{valid_geo['density'].mean():.4f}")
            k3.metric("Max Density", f"{valid_geo['density'].max():.4f}")

            st.subheader("Top 10 Highest Density Locations")
            top_density = valid_geo.nlargest(10, "density")[
                ["camera_id", "camera_label", "avg_traffic", "density", "records"]
            ].reset_index(drop=True)
            top_density["density"] = top_density["density"].apply(lambda x: f"{x:.4f}")
            top_density.index = range(1, len(top_density) + 1)
            top_density.index.name = "Rank"
            st.dataframe(top_density, width="stretch")
        except Exception as e:
            st.error(f"Error in KDE analysis: {e}")

with tab_external:
    st.subheader("1. Weather & Calendar Effects (ERA5 Weather & Thai Holidays)")
    ext_df = df.copy()
    ext_df["metric_val"] = pd.to_numeric(ext_df[selected_metric], errors="coerce").fillna(0.0)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if "day_type" in ext_df.columns and ext_df["day_type"].nunique() > 1:
            day_hourly = ext_df.groupby(["hour", "day_type"], as_index=False)["metric_val"].mean()
            fig_day = px.line(
                day_hourly,
                x="hour",
                y="metric_val",
                color="day_type",
                markers=True,
                title="Hourly Traffic by Day Type (Weekday vs Weekend / Holiday)",
                labels={"hour": "Hour", "metric_val": "Mean Count", "day_type": "Day Type"},
            )
            st.plotly_chart(fig_day, width="stretch")
        elif "day_of_week" in ext_df.columns:
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_df = ext_df.groupby("day_of_week", as_index=False)["metric_val"].mean()
            fig_dow = px.bar(
                dow_df,
                x="day_of_week",
                y="metric_val",
                category_orders={"day_of_week": dow_order},
                title="Mean Traffic by Day of Week",
                labels={"day_of_week": "Day of Week", "metric_val": "Mean Count"},
            )
            st.plotly_chart(fig_dow, width="stretch")

    with col_e2:
        if "rain_group" in ext_df.columns and ext_df["rain_group"].nunique() > 1:
            rain_summary = ext_df.groupby("rain_group", as_index=False).agg(
                mean_count=("metric_val", "mean"),
                frames=("metric_val", "count"),
            )
            fig_rain = px.bar(
                rain_summary,
                x="rain_group",
                y="mean_count",
                color="rain_group",
                hover_data=["frames"],
                title="Mean Traffic by Rain Condition (ERA5)",
                labels={"rain_group": "Rain Group", "mean_count": "Mean Count"},
            )
            fig_rain.update_layout(showlegend=False)
            st.plotly_chart(fig_rain, width="stretch")
        elif "temperature_2m_c" in ext_df.columns:
            ext_df["temp_bin"] = ext_df["temperature_2m_c"].round(0)
            temp_df = ext_df.groupby("temp_bin", as_index=False)["metric_val"].mean()
            fig_temp = px.line(
                temp_df,
                x="temp_bin",
                y="metric_val",
                markers=True,
                title="Mean Traffic vs Temperature (C)",
                labels={"temp_bin": "Temperature (C)", "metric_val": "Mean Count"},
            )
            st.plotly_chart(fig_temp, width="stretch")

    st.divider()
    st.subheader("2. Road Network Context (OpenStreetMap)")
    col_osm1, col_osm2 = st.columns(2)

    with col_osm1:
        if "road_class_300m" in ext_df.columns:
            road_df = (
                ext_df.groupby("road_class_300m", as_index=False)
                .agg(mean_count=("metric_val", "mean"), cameras=("camera_id", "nunique"))
                .sort_values("mean_count", ascending=False)
            )
            fig_road = px.bar(
                road_df,
                x="road_class_300m",
                y="mean_count",
                color="mean_count",
                color_continuous_scale="Blues",
                hover_data=["cameras"],
                title="Mean Traffic by Primary OSM Road Class (300m Radius)",
                labels={"road_class_300m": "Road Class", "mean_count": "Mean Count"},
            )
            fig_road.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_road, width="stretch")

    with col_osm2:
        if "osm" in extras and {"camera_id", "lanes_num"}.issubset(extras["osm"].columns):
            lanes_per_cam = (
                extras["osm"]
                .dropna(subset=["lanes_num"])
                .groupby("camera_id", as_index=False)["lanes_num"]
                .max()
            )
            cam_lanes = cam_summary.merge(lanes_per_cam, on="camera_id", how="inner")
            if not cam_lanes.empty:
                lanes_agg = (
                    cam_lanes.groupby("lanes_num", as_index=False)
                    .agg(mean_count=("avg_traffic", "mean"), cameras=("camera_id", "count"))
                    .sort_values("lanes_num")
                )
                lanes_agg["lanes_label"] = lanes_agg["lanes_num"].astype(int).astype(str) + " lanes"
                fig_lanes = px.bar(
                    lanes_agg,
                    x="lanes_label",
                    y="mean_count",
                    hover_data=["cameras"],
                    title="Mean Traffic by Max Road Lanes near Camera (OSM)",
                    labels={"lanes_label": "Max Lanes", "mean_count": "Mean Count"},
                )
                st.plotly_chart(fig_lanes, width="stretch")

    st.divider()
    st.subheader("3. Traffic Accidents (MOT 2026) & BEM Expressway Volume")
    col_acc, col_bem = st.columns(2)

    with col_acc:
        if "accidents" in extras and "suspected_cause" in extras["accidents"].columns:
            acc_df = extras["accidents"]
            cause_df = (
                acc_df["suspected_cause"]
                .dropna()
                .value_counts()
                .head(8)
                .reset_index()
            )
            cause_df.columns = ["suspected_cause", "count"]
            fig_cause = px.bar(
                cause_df,
                x="count",
                y="suspected_cause",
                orientation="h",
                title="Top Suspected Accident Causes in Bangkok (2026)",
                labels={"count": "Reported Accidents", "suspected_cause": "Suspected Cause"},
            )
            fig_cause.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_cause, width="stretch")
        elif "mot_reports_300m_hour" in ext_df.columns:
            ext_df["has_incident"] = np.where(ext_df["mot_reports_300m_hour"] > 0, "Nearby Incident", "No Incident")
            inc_df = ext_df.groupby("has_incident", as_index=False)["metric_val"].mean()
            fig_inc = px.bar(
                inc_df,
                x="has_incident",
                y="metric_val",
                title="Mean Traffic: Nearby Incident vs Normal",
                labels={"has_incident": "Incident Status", "metric_val": "Mean Count"},
            )
            st.plotly_chart(fig_inc, width="stretch")

    with col_bem:
        if "bem" in extras and {"year", "month", "traffic_volume"}.issubset(extras["bem"].columns):
            bem_df = extras["bem"].copy()
            if "measure" in bem_df.columns and bem_df["measure"].nunique() > 1:
                primary_measure = bem_df["measure"].iloc[0]
                bem_df = bem_df[bem_df["measure"] == primary_measure]
            recent_bem = bem_df[bem_df["year"] >= bem_df["year"].max() - 3].copy()
            recent_bem["year_str"] = recent_bem["year"].astype(str)
            fig_bem = px.line(
                recent_bem.sort_values(["year", "month"]),
                x="month",
                y="traffic_volume",
                color="year_str",
                markers=True,
                title="BEM Expressway Monthly Traffic Volume (Recent Years)",
                labels={"month": "Month (1 - 12)", "traffic_volume": "Traffic Volume", "year_str": "Year"},
            )
            fig_bem.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_bem, width="stretch")

    st.divider()
    st.subheader("Filtered Camera Summary Table")
    st.dataframe(filtered_cams, width="stretch")
    csv_bytes = filtered_cams.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download Summary CSV", data=csv_bytes, file_name="bangkok_cctv_summary.csv", mime="text/csv")

with tab_model:
    st.subheader("Object Detection Model & Speed Evaluation")

    if "speed_benchmark" in extras:
        sb_df = extras["speed_benchmark"]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if {"model", "img_per_sec"}.issubset(sb_df.columns):
                fig_fps = px.bar(
                    sb_df,
                    x="model",
                    y="img_per_sec",
                    color="model",
                    title="Inference Throughput (Images / Second)",
                    labels={"model": "Model", "img_per_sec": "Images / sec"},
                )
                fig_fps.update_layout(showlegend=False)
                st.plotly_chart(fig_fps, width="stretch")
        with col_m2:
            if {"model", "ms_predict", "ms_read"}.issubset(sb_df.columns):
                sb_melt = sb_df.melt(
                    id_vars="model",
                    value_vars=["ms_read", "ms_predict"],
                    var_name="stage",
                    value_name="milliseconds",
                )
                fig_ms = px.bar(
                    sb_melt,
                    x="model",
                    y="milliseconds",
                    color="stage",
                    title="Latency Breakdown per Image (ms)",
                    labels={"model": "Model", "milliseconds": "Latency (ms)", "stage": "Stage"},
                )
                st.plotly_chart(fig_ms, width="stretch")

        st.markdown("**Speed Benchmark Table (61k Full Dataset Projection)**")
        st.dataframe(sb_df, width="stretch")

    if "model_metrics" in extras:
        st.divider()
        st.markdown("**Counting Accuracy Metrics (MAE / RMSE)**")
        st.dataframe(extras["model_metrics"], width="stretch")

    if "model_charts" in extras:
        st.divider()
        st.markdown("**Validation Charts (Day vs Night & Vehicle Class Error)**")
        chart_cols = st.columns(len(extras["model_charts"]))
        for idx, img_path in enumerate(extras["model_charts"]):
            with chart_cols[idx]:
                st.image(img_path, width="stretch")