import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import warnings
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
import datetime
from scipy.spatial import cKDTree

# --- KONFIGURACJA ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="eZagrożeNIE", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #F6F6EA; }
        .stSpinner > div > div { color: black !important; }
    </style>
    """,
    unsafe_allow_html=True
)

MAX_DISTANCE_DEGREES = 0.001
ROUTE_SIMPLIFICATION_TOLERANCE = 0.0001  # Simplify route geometry

# --- OPTIMIZED DATA LOADING WITH SPATIAL INDEX ---
@st.cache_data
def load_all_data():
    df = pd.read_csv("dane_wypadki_2018_2024.csv")
    # Pre-convert timestamps to datetime for faster repeated access
    if 'unix_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['weekday'] = df['datetime'].dt.weekday + 1
    return df

@st.cache_data
def build_spatial_index():
    """Build KD-tree for ultra-fast spatial queries."""
    coords = all_data[['gps_y_dec', 'gps_x_dec']].values
    return cKDTree(coords)

all_data = load_all_data()
spatial_index = build_spatial_index()

# --- OPTIMIZED STREET FUNCTIONS ---
def fetch_street_suggestions(query: str) -> list[str]:
    if not query or len(query) < 3: 
        return []
    try:
        mask = all_data['ulica'].str.contains(query, case=False, na=False)
        rows = all_data[mask][['ulica', 'miejscowosc']].drop_duplicates().head(10)
        return [f"{u}, {m}" for u, m in zip(rows['ulica'], rows['miejscowosc']) if u]
    except: 
        return []

def street_to_coords(street_city: str) -> Optional[Tuple[float, float]]:
    try:
        street, city = street_city.split(", ", 1)
        mask = (all_data['ulica'] == street) & (all_data['miejscowosc'] == city)
        row = all_data[mask]
        if not row.empty:
            return (float(row['gps_y_dec'].mean()), float(row['gps_x_dec'].mean()))
        return None
    except: 
        return None

# --- ROUTE SIMPLIFICATION (RAMER-DOUGLAS-PEUCKER) ---
def simplify_route(coords: List[Tuple[float, float]], tolerance: float = ROUTE_SIMPLIFICATION_TOLERANCE) -> List[Tuple[float, float]]:
    """Simplify route to reduce number of points while preserving shape."""
    if len(coords) <= 2:
        return coords
    
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line."""
        if line_start == line_end:
            return np.linalg.norm(np.array(point) - np.array(line_start))
        
        p = np.array(point)
        s = np.array(line_start)
        e = np.array(line_end)
        
        line_vec = e - s
        point_vec = p - s
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        
        if t < 0.0:
            return np.linalg.norm(point_vec)
        elif t > 1.0:
            return np.linalg.norm(p - e)
        
        nearest = s + t * line_vec
        return np.linalg.norm(p - nearest)
    
    def rdp(points, epsilon):
        """Ramer-Douglas-Peucker algorithm."""
        dmax = 0.0
        index = 0
        
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        
        if dmax > epsilon:
            left = rdp(points[:index + 1], epsilon)
            right = rdp(points[index:], epsilon)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]
    
    return rdp(coords, tolerance)

# --- ULTRA-FAST SPATIAL QUERY USING KD-TREE ---
def fetch_accidents_fast(route_coords):
    """Ultra-fast accident fetching using spatial index."""
    if not route_coords:
        return pd.DataFrame()
    
    # Simplify route first to reduce computation
    simplified_coords = simplify_route(route_coords)
    
    # Query spatial index with all route points at once
    route_points = np.array(simplified_coords)
    
    # Find all accidents within distance threshold of any route point
    # Using KD-tree query_ball_point for ultra-fast spatial search
    search_radius = MAX_DISTANCE_DEGREES * 1.5  # Slightly larger to catch all
    
    # Query all route points at once
    nearby_indices_list = spatial_index.query_ball_point(route_points, search_radius)
    
    # Flatten and get unique indices
    nearby_indices = set()
    for indices in nearby_indices_list:
        nearby_indices.update(indices)
    
    if not nearby_indices:
        return pd.DataFrame()
    
    # Get accidents from indices
    nearby_indices = list(nearby_indices)
    df = all_data.iloc[nearby_indices].copy()
    
    # Refine with exact distance calculation (vectorized)
    accident_points = df[['gps_y_dec', 'gps_x_dec']].values
    
    # Calculate minimum distance to any route segment
    min_distances = np.full(len(df), np.inf)
    
    for i in range(len(simplified_coords) - 1):
        distances = vectorized_point_to_segment_distances(
            accident_points,
            simplified_coords[i],
            simplified_coords[i + 1]
        )
        min_distances = np.minimum(min_distances, distances)
    
    # Filter by exact threshold
    mask = min_distances <= MAX_DISTANCE_DEGREES
    result_df = df[mask].copy()
    
    if not result_df.empty:
        result_df = result_df.rename(columns={'gps_y_dec': 'lat', 'gps_x_dec': 'lon'})
        result_df["Odległość [m]"] = (min_distances[mask] * 111000).round(1)
        return result_df.drop_duplicates(subset=["id"])
    
    return pd.DataFrame()

# --- VECTORIZED GEOMETRY FUNCTIONS ---
def vectorized_point_to_segment_distances(points: np.ndarray, segment_start: Tuple[float, float], 
                                         segment_end: Tuple[float, float]) -> np.ndarray:
    """Vectorized distance calculation for all points to a segment."""
    P = points
    A = np.array(segment_start)
    B = np.array(segment_end)
    
    AB = B - A
    denom = np.dot(AB, AB)
    
    if denom == 0:
        return np.linalg.norm(P - A, axis=1)
    
    t = np.clip(np.dot(P - A, AB) / denom, 0, 1)
    closest = A + t[:, np.newaxis] * AB
    return np.linalg.norm(P - closest, axis=1)

# --- OSRM ROUTING WITH SESSION REUSE ---
@st.cache_data(show_spinner=False)
def get_osrm_routes(start, end, via=None):
    session = requests.Session()
    
    def call_osrm(p1, p2):
        url = f"http://router.project-osrm.org/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=geojson&alternatives=true"
        try:
            r = session.get(url, timeout=8).json()
            return r.get("routes", [])
        except: 
            return []
    
    all_routes = []
    if not via:
        routes = call_osrm(start, end)
        for i, r in enumerate(routes):
            all_routes.append({
                "label": "Trasa główna" if i == 0 else f"Alternatywa {i}",
                "coords": [(lat, lon) for lon, lat in r["geometry"]["coordinates"]],
                "distance": round(r["distance"]/1000, 2),
                "duration": round(r["duration"]/60, 1)
            })
    else:
        routes_a = call_osrm(start, via)
        routes_b = call_osrm(via, end)
        idx = 0
        for i, ra in enumerate(routes_a[:2]):
            for j, rb in enumerate(routes_b[:2]):
                c_a = [(lat, lon) for lon, lat in ra["geometry"]["coordinates"]]
                c_b = [(lat, lon) for lon, lat in rb["geometry"]["coordinates"]]
                all_routes.append({
                    "label": f"Alternatywna trasa {idx + 1}",
                    "coords": c_a[:-1] + c_b,
                    "distance": round((ra["distance"] + rb["distance"])/1000, 2),
                    "duration": round((ra["duration"] + rb["duration"])/60, 1)
                })
                idx += 1
                if idx >= 4: break
            if idx >= 4: break
    return all_routes

# --- OPTIMIZED REGRESSION CALCULATION ---
@st.cache_data(show_spinner=False)
def calculate_probability(accidents_df):
    """Optimized probability calculation with vectorized operations."""
    if accidents_df.empty:
        return None
    
    # Use pre-computed time columns
    final_dataframe = accidents_df[['lon', 'hour', 'day', 'month', 'year', 'weekday']].copy()
    
    # Vectorized time categorization
    final_dataframe['time_of_day'] = pd.cut(
        final_dataframe['hour'],
        bins=[-1, 5, 8, 13, 17, 24],
        labels=['0-6', '6-9', '9-14', '14-18', '18-24']
    ).astype(str)
    
    final_dataframe.drop(['hour'], axis=1, inplace=True)
    
    # Generate zero observations efficiently
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    time_categories = ['0-6', '6-9', '9-14', '14-18', '18-24']
    
    zero_rows = []
    for month_idx, month_len in enumerate(month_lengths):
        for day in range(1, month_len + 1):
            for time_cat in time_categories:
                zero_rows.append({
                    'lon': None,
                    'time_of_day': time_cat,
                    'day': day,
                    'month': month_idx + 1,
                    'year': 2024
                })
    
    zero_df = pd.DataFrame(zero_rows)
    
    # Combine and deduplicate
    final_dataframe = pd.concat([final_dataframe, zero_df], ignore_index=True)
    final_dataframe.drop_duplicates(subset=['time_of_day', 'day', 'month', 'year'], 
                                    keep='first', inplace=True, ignore_index=True)
    
    # Vectorized accident indicator
    final_dataframe['accidents'] = final_dataframe['lon'].notna().astype(int)
    final_dataframe.drop('lon', axis=1, inplace=True)
    
    # Vectorized weekday calculation
    try:
        dates = pd.to_datetime(final_dataframe[['year', 'month', 'day']])
        final_dataframe['weekday'] = dates.dt.weekday + 1
    except:
        final_dataframe['weekday'] = [
            datetime.date(int(row['year']), int(row['month']), int(row['day'])).weekday() + 1 
            for _, row in final_dataframe.iterrows()
        ]
    
    final_dataframe.drop(['day', 'year'], axis=1, inplace=True)
    
    # Create dummy variables
    final_dataframe = pd.get_dummies(final_dataframe, columns=['time_of_day', 'weekday', 'month'])
    
    # Create interactions efficiently
    time_cols = [col for col in final_dataframe.columns if col.startswith('time_of_day_')]
    weekday_cols = [col for col in final_dataframe.columns if col.startswith('weekday_')]
    
    for time_col in time_cols:
        for weekday_col in weekday_cols:
            final_dataframe[f'{weekday_col}*{time_col}'] = \
                final_dataframe[weekday_col] * final_dataframe[time_col]
    
    # Fit model
    X = final_dataframe.drop('accidents', axis=1)
    y = final_dataframe['accidents']
    
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X, y)
    
    # Current time prediction
    now = datetime.datetime.now()
    time_of_day = '0-6' if now.hour < 6 else '6-9' if now.hour < 9 else '9-14' if now.hour < 14 else '14-18' if now.hour < 18 else '18-24'
    
    pred_dataframe = pd.DataFrame(0, index=[0], columns=X.columns)
    
    time_col = f'time_of_day_{time_of_day}'
    weekday_col = f'weekday_{now.weekday() + 1}'
    month_col = f'month_{now.month}'
    interaction_col = f'{weekday_col}*{time_col}'
    
    for col in [time_col, weekday_col, month_col, interaction_col]:
        if col in pred_dataframe.columns:
            pred_dataframe[col] = 1
    
    prob_val = round(model.predict_proba(pred_dataframe)[0][1] * 100, 2)
    
    return time_of_day, prob_val

# --- SESSION STATE INITIALIZATION ---
if "base_result" not in st.session_state: st.session_state.base_result = None
if "result" not in st.session_state: st.session_state.result = None
if "via_point" not in st.session_state: st.session_state.via_point = None
if "pending_point" not in st.session_state: st.session_state.pending_point = None
if "show_table" not in st.session_state: st.session_state.show_table = False
if "selected_route_idx" not in st.session_state: st.session_state.selected_route_idx = 0
if "accidents_loaded" not in st.session_state: st.session_state.accidents_loaded = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("logo transparent.png", use_container_width=True)
    st.markdown("---")
    st.header("Parametry Trasy")
    
    def input_point(label, key):
        m = st.radio(f"{label}", ["Ulica", "Współrzędne"], key=f"m_{key}")
        if m == "Ulica":
            q = st.text_input(f"Wpisz nazwę ulicy ({label})", key=f"q_{key}")
            sug = fetch_street_suggestions(q)
            sel = st.selectbox(f"Wybierz z listy", sug, key=f"s_{key}")
            return street_to_coords(sel) if sel else None
        else:
            raw = st.text_input(f"Lat, Lon ({label})", 
                              "51.938, 15.513" if key=="st" else "51.936, 15.506", 
                              key=f"r_{key}")
            try: 
                return tuple(map(float, raw.split(",")))
            except: 
                return None

    start_pt = input_point("Punkt startowy", "st")
    end_pt = input_point("Punkt docelowy", "en")

    if st.button("Wyznacz trasę", use_container_width=True):
        if start_pt and end_pt:
            st.session_state.via_point = None
            st.session_state.pending_point = None
            st.session_state.accidents_loaded = False
            
            # INSTANT ROUTE DISPLAY - Get routes immediately without accidents
            with st.spinner("Wyznaczanie tras..."):
                routes = get_osrm_routes(start_pt, end_pt)
                
                # Store routes WITHOUT accidents first (instant display)
                results = []
                for r in routes:
                    r["acc"] = pd.DataFrame()  # Empty for now
                    r["acc_count"] = 0
                    results.append(r)
                
                st.session_state.result = results
                st.session_state.base_result = results        
                st.session_state.selected_route_idx = 0
                st.session_state.start_pt = start_pt
                st.session_state.end_pt = end_pt
        else: 
            st.error("Proszę poprawnie określić punkty!")

# --- MAIN RESULTS ---
if st.session_state.result:
    routes = st.session_state.result

    # Load accidents progressively AFTER displaying routes
    if not st.session_state.accidents_loaded:
        with st.spinner("Ładowanie danych o wypadkach..."):
            for r in routes:
                acc_df = fetch_accidents_fast(r["coords"])
                r["acc"] = acc_df
                r["acc_count"] = len(acc_df)
            st.session_state.result = routes
            st.session_state.accidents_loaded = True
            st.rerun()

    cols = st.columns(len(routes))
    for i, r in enumerate(routes):
        with cols[i]:
            is_selected = (st.session_state.selected_route_idx == i)
            label = f"{r['label']}: {r['acc_count']} wypadków"
            if st.button(label, key=f"sel_{i}", use_container_width=True, 
                        type="primary" if is_selected else "secondary"):
                st.session_state.selected_route_idx = i
                st.rerun()

    res = routes[st.session_state.selected_route_idx]

    if st.session_state.via_point and st.session_state.base_result:
        if st.button("↩ Wróć do trasy głównej", type="secondary"):
            st.session_state.result = st.session_state.base_result
            st.session_state.selected_route_idx = 0
            st.session_state.via_point = None
            st.session_state.pending_point = None
            st.session_state.accidents_loaded = True
            st.rerun()

    st.markdown(
        f"<p style='color:black; font-weight:bold;'>"
        f"Dystans: {res['distance']} km | Czas: {res['duration']} min | Zdarzenia: {res['acc_count']}"
        f"</p>",
        unsafe_allow_html=True
    )
    
    m = folium.Map(location=st.session_state.start_pt, zoom_start=12, prefer_canvas=True)
    for i, r in enumerate(routes):
        folium.PolyLine(r["coords"], 
                       color="#A9A9A9" if i != st.session_state.selected_route_idx else "#2A75BB", 
                       weight=4 if i != st.session_state.selected_route_idx else 8, 
                       opacity=0.7).add_to(m)

    folium.Marker(st.session_state.start_pt, 
                 icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)
    folium.Marker(st.session_state.end_pt, 
                 icon=folium.Icon(color="red", icon="stop", prefix='fa')).add_to(m)
    
    display_orange = st.session_state.via_point if st.session_state.via_point else st.session_state.pending_point
    if display_orange:
        folium.Marker(display_orange, 
                     icon=folium.Icon(color="orange", icon="map-pin")).add_to(m)

    if not res["acc"].empty:
        cluster = MarkerCluster(options={'maxClusterRadius': 50}).add_to(m)
        for _, row in res["acc"].iterrows():
            folium.CircleMarker((row["lat"], row["lon"]), radius=5, color="red", fill=True, 
                              tooltip=(
                f"Wypadek<br>"
                f"Miejscowość: {row['miejscowosc']}<br>"
                f"Ulica: {row['ulica']}<br>"
                f"Odległość od trasy: {row['Odległość [m]']} m"
            )).add_to(cluster)

    map_data = st_folium(m, key="main_map", height=500, width="100%", 
                        returned_objects=["last_clicked"])
    
    if map_data and map_data.get("last_clicked"):
        clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        if st.session_state.pending_point != clicked:
            st.session_state.pending_point = clicked
            st.rerun()

    if st.session_state.pending_point:
        st.markdown(f"<p style='color:black; font-weight:bold;'>📍 Wybrano punkt: {st.session_state.pending_point}</p>", 
                   unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Wyznacz trasę przez punkt", type="primary"):
                if st.session_state.base_result is None:
                    st.session_state.base_result = st.session_state.result
                st.session_state.via_point = st.session_state.pending_point
                st.session_state.accidents_loaded = False
                
                with st.spinner("Wyznaczanie nowej trasy..."):
                    new_routes = get_osrm_routes(st.session_state.start_pt, 
                                                st.session_state.end_pt, 
                                                st.session_state.via_point)
                    res_new = []
                    for r in new_routes:
                        r["acc"] = pd.DataFrame()
                        r["acc_count"] = 0
                        res_new.append(r)
                    st.session_state.result = res_new
                    st.session_state.selected_route_idx = 0
                    st.session_state.pending_point = None
                    st.rerun()
        with col2:
            if st.button("❌ Usuń punkt"):
                st.session_state.pending_point = None
                st.rerun()

    if not res["acc"].empty:
        if st.button("Pokaż szczegóły zdarzeń"):
            st.session_state.show_table = not st.session_state.show_table
        if st.session_state.show_table:
            st.dataframe(res["acc"][["miejscowosc", "ulica", "Odległość [m]"]], 
                        use_container_width=True, hide_index=True)

    # --- REGRESSION SECTION ---
    if not res["acc"].empty:
        result = calculate_probability(res["acc"])
        if result:
            time_of_day, prob_val = result
            st.markdown(
                f"<p style='color: black; font-weight: bold; font-size: 1.1em;'>"
                f"Prawdopodobieństwo wypadku w godzinach {time_of_day}: {prob_val}%"
                f"</p>", 
                unsafe_allow_html=True
            )

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: 
        st.image("pl.gif", use_container_width=True)
