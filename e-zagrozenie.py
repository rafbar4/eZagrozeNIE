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

MAX_DISTANCE_DEGREES = 0.001  # ok. 100 m


@st.cache_data
def load_all_data():
    # Plik musi nazywać się dokładnie tak i być w tym samym folderze na GitHub
    df = pd.read_csv("dane_wypadki_export_2024.csv")
    return df

# Wczytujemy dane raz na starcie
all_data = load_all_data()

def fetch_street_suggestions(query: str) -> list[str]:
    if not query or len(query) < 3: return []
    try:
        # Filtrowanie w pandas zastępuje SQL ILIKE
        mask = all_data['ulica'].str.contains(query, case=False, na=False)
        rows = all_data[mask][['ulica', 'miejscowosc']].drop_duplicates().head(10)
        return [f"{u}, {m}" for u, m in zip(rows['ulica'], rows['miejscowosc']) if u]
    except: return []

def street_to_coords(street_city: str) -> Optional[Tuple[float, float]]:
    try:
        street, city = street_city.split(", ", 1)
        mask = (all_data['ulica'] == street) & (all_data['miejscowosc'] == city)
        row = all_data[mask]
        if not row.empty:
            return (float(row['gps_y_dec'].mean()), float(row['gps_x_dec'].mean()))
        return None
    except: return None

def fetch_accidents(route_coords):
    lats, lons = [p[0] for p in route_coords], [p[1] for p in route_coords]
    # Filtrowanie zakresu zastępuje SQL BETWEEN
    mask = (all_data['gps_y_dec'].between(min(lats)-0.01, max(lats)+0.01)) & \
           (all_data['gps_x_dec'].between(min(lons)-0.01, max(lons)+0.01))
    
    df = all_data[mask].copy()
    # Zmiana nazw na potrzeby Twojej dalszej logiki
    df.rename(columns={'gps_y_dec': 'lat', 'gps_x_dec': 'lon'}, inplace=True)
    
    if df.empty: return pd.DataFrame()
    
    all_hits = []
    for i in range(len(route_coords)-1):
        A, B = route_coords[i], route_coords[i+1]
        df_c = df.copy()
        df_c["d"] = df_c.apply(lambda r: point_to_segment_distance(r["lat"], r["lon"], A[0], A[1], B[0], B[1]), axis=1)
        hits = df_c[df_c["d"] <= MAX_DISTANCE_DEGREES]
        if not hits.empty: all_hits.append(hits)
    
    if all_hits:
        res_df = pd.concat(all_hits).drop_duplicates(subset=["id"])
        res_df["Odległość [m]"] = round(res_df["d"] * 111000, 1)
        return res_df
    return pd.DataFrame()


def point_to_segment_distance(px, py, ax, ay, bx, by):
    P, A, B = np.array([px, py]), np.array([ax, ay]), np.array([bx, by])
    AB = B - A
    denom = np.dot(AB, AB)
    t = max(0, min(1, np.dot(P - A, AB) / (denom if denom != 0 else 1)))
    closest = A + t * AB
    return np.linalg.norm(P - closest)

@st.cache_data(show_spinner=False)
def get_osrm_routes(start, end, via=None):
    def call_osrm(p1, p2):
        url = f"http://router.project-osrm.org/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=geojson&alternatives=true"
        try:
            r = requests.get(url, timeout=5).json()
            return r.get("routes", [])
        except: return []
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

# --- INICJALIZACJA SESJI ---
if "base_result" not in st.session_state: st.session_state.base_result = None
if "result" not in st.session_state: st.session_state.result = None
if "via_point" not in st.session_state: st.session_state.via_point = None
if "pending_point" not in st.session_state: st.session_state.pending_point = None
if "show_table" not in st.session_state: st.session_state.show_table = False
if "selected_route_idx" not in st.session_state: st.session_state.selected_route_idx = 0

# --- SIDEBAR & RESZTA INTERFEJSU ---
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
            raw = st.text_input(f"Lat, Lon ({label})", "51.938, 15.513" if key=="st" else "51.936, 15.506", key=f"r_{key}")
            try: return tuple(map(float, raw.split(",")))
            except: return None

    start_pt = input_point("Punkt startowy", "st")
    end_pt = input_point("Punkt docelowy", "en")

    if st.button("Wyznacz trasę", use_container_width=True):
        if start_pt and end_pt:
            st.session_state.via_point = None
            st.session_state.pending_point = None
            with st.spinner("Analizowanie tras..."):
                routes = get_osrm_routes(start_pt, end_pt)
                results = []
                for r in routes:
                    acc_df = fetch_accidents(r["coords"])
                    r["acc"] = acc_df
                    r["acc_count"] = len(acc_df)
                    results.append(r)
                st.session_state.result = results
                st.session_state.base_result = results        
                st.session_state.selected_route_idx = 0
                st.session_state.start_pt = start_pt
                st.session_state.end_pt = end_pt
        else: st.error("Proszę poprawnie określić punkty!")


# --- WYNIKI ---
if st.session_state.result:
    routes = st.session_state.result  # ...a teraz są tu

    # Podświetlanie wybranego przycisku
    cols = st.columns(len(routes))
    for i, r in enumerate(routes):
        with cols[i]:
            is_selected = (st.session_state.selected_route_idx == i)
            label = f"{r['label']}: {r['acc_count']} wypadków"
            if st.button(label, key=f"sel_{i}", width='stretch', type="primary" if is_selected else "secondary"):
                st.session_state.selected_route_idx = i
                st.rerun()

    res = routes[st.session_state.selected_route_idx]  # a tu jest tylko ta trasa (ze wszystkimi swoimi kluczami) która została wybrana za pomocą "selected_route_idx" (to jest kolejnośc w liście)

    if st.session_state.via_point and st.session_state.base_result:
        if st.button("↩ Wróć do trasy głównej", type="secondary"):
            st.session_state.result = st.session_state.base_result
            st.session_state.selected_route_idx = 0
            st.session_state.via_point = None
            st.session_state.pending_point = None
            st.rerun()

    st.markdown(
        f"<p style='color:black; font-weight:bold;'>"
        f"Dystans: {res['distance']} km | Czas: {res['duration']} min | Zdarzenia: {res['acc_count']}"
        f"</p>",
        unsafe_allow_html=True
    )
    
    m = folium.Map(location=st.session_state.start_pt, zoom_start=12)
    for i, r in enumerate(routes):
        folium.PolyLine(r["coords"], color="#A9A9A9" if i != st.session_state.selected_route_idx else "#2A75BB", 
                        weight=4 if i != st.session_state.selected_route_idx else 8, opacity=0.7).add_to(m)

    folium.Marker(st.session_state.start_pt, icon=folium.Icon(color="green", icon="play", prefix='fa')).add_to(m)
    folium.Marker(st.session_state.end_pt, icon=folium.Icon(color="red", icon="stop", prefix='fa')).add_to(m)
    
    # Punkt 3: Pokazujemy pomarańczowy znacznik także dla "pending_point" (klikniętego, ale jeszcze nieprzeliczonego)
    display_orange = st.session_state.via_point if st.session_state.via_point else st.session_state.pending_point
    if display_orange:
        folium.Marker(display_orange, icon=folium.Icon(color="orange", icon="map-pin")).add_to(m)

    if not res["acc"].empty:
        cluster = MarkerCluster().add_to(m)
        for _, row in res["acc"].iterrows():
            folium.CircleMarker((row["lat"], row["lon"]), radius=5, color="red", fill=True, 
                                tooltip=(
            f"Wypadek<br>"
            f"Miejscowość: {row['miejscowosc']}<br>"
            f"Ulica: {row['ulica']}<br>"
            f"Odległość od trasy: {row['Odległość [m]']} m"
            )
        ).add_to(cluster)

    map_data = st_folium(m, key="main_map", height=500, width="100%", returned_objects=["last_clicked"])
    
    # Obsługa kliknięć wewnątrz bloku wyników
    if map_data and map_data.get("last_clicked"):
        clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        if st.session_state.pending_point != clicked:
            st.session_state.pending_point = clicked
            st.rerun()

    if st.session_state.pending_point:
        st.markdown(f"<p style='color:black; font-weight:bold;'>📍 Wybrano punkt: {st.session_state.pending_point}</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Wyznacz trasę przez punkt", type="primary"):
                if st.session_state.base_result is None:
                    st.session_state.base_result = st.session_state.result
                st.session_state.via_point = st.session_state.pending_point
                with st.spinner("Analizuję i przeliczam trasę..."):
                    new_routes = get_osrm_routes(st.session_state.start_pt, st.session_state.end_pt, st.session_state.via_point)
                    res_new = []
                    for r in new_routes:
                        acc = fetch_accidents(r["coords"])
                        r["acc"] = acc
                        r["acc_count"] = len(acc)
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
            st.dataframe(res["acc"][["miejscowosc", "ulica", "Odległość [m]"]], use_container_width=True, hide_index=True)

# --- SEKCJA REGRESJI ---

    # tu robię dataframe specjalnie do regresji:
        final_dataframe = res['acc'][['lon', 'unix_time']].copy()
    # klumna "gps_y_dec" jest niepotrzebna bo z perspektywy zero-jedynkowych obserwacji zawiera taką sama informację jak "gps_x_dec"
        final_dataframe['time_of_day'] = ['0-6' if datetime.datetime.fromtimestamp(i).hour in range(0, 6) else '6-9' if datetime.datetime.fromtimestamp(i).hour in range(6, 9) else '9-14' if datetime.datetime.fromtimestamp(i).hour in range(9, 14) else '14-18' if datetime.datetime.fromtimestamp(i).hour in range(14, 18) else '18-24' for i in final_dataframe['unix_time']]
    
    # tworzenie kolumny "pora dnia":
    
    # kolumna "dzień":
        final_dataframe['day'] = [datetime.datetime.fromtimestamp(i).day for i in final_dataframe['unix_time']]
    
    # kolumna "miesiąc": 
        final_dataframe['month'] = [datetime.datetime.fromtimestamp(i).month for i in final_dataframe['unix_time']]
    # uwaga, różne miesiące mają różną liczbę poszczególnych dni tygodnia, np. jesli w jakimś miesiącu jest pięć wtorków a w innym tylko cztery to prawdopodobieństwo wypadku we wtorek w tym pierwszym miesiącu wyjdzie wyższe niż w tym drugim po zagregowaniu według dnia tygodnia mimo że wcale tak nie jest       
    
    # kolumna "rok": 
        final_dataframe['year'] = [datetime.datetime.fromtimestamp(i).year for i in final_dataframe['unix_time']]
    
    # czas uniksowy już nie będzie potrzebny:
        final_dataframe.drop('unix_time', axis='columns', inplace=True)

    # długości kolejnych miesięcy do dorabiania zerowych obserwacji:
        month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # dorabianie zerowych obserwacji w brakujących jednostkach czasu:
        for i in range(len(month_lengths)):
            final_dataframe = pd.concat([final_dataframe, pd.DataFrame({'lon': [None] * month_lengths[i] * 5, 'time_of_day': ['0-6'] * month_lengths[i] + ['6-9'] * month_lengths[i] + ['9-14'] * month_lengths[i] + ['14-18'] * month_lengths[i] + ['18-24'] * month_lengths[i], 'day': [x for x in range(1, month_lengths[i] + 1)] * 5, 'month': [i + 1] * month_lengths[i] * 5, 'year': [2024] * month_lengths[i] * 5})], ignore_index=True)
    
    # usuwanie duplikatów:
        final_dataframe.drop_duplicates(subset=['time_of_day', 'day', 'month', 'year'], keep='first', inplace=True, ignore_index=True)
    # jeśli jest w danej jednostce czasu (przedział godzinowy danego dnia) jakiś wypadek to obserwacja zerowa z końca datafrejmu się usunie, również wiele wypadków w tej samej jednostce czasu się usunie i zostanie jeden dzięki czemu będzie można zrobić zero-jedynkowe obserwacje
        final_dataframe['lon'] = [0 if str(x) == 'nan' else 1 for x in final_dataframe['lon']]  # dychotomizujemy wypadki
        final_dataframe.rename(columns={'lon': 'accidents'}, inplace=True) # zmieniamy nazwę kolumny
    # kolumna "dzień tygodnia":
        final_dataframe['weekday'] = [datetime.date(int(final_dataframe['year'][i]), int(final_dataframe['month'][i]), int(final_dataframe['day'][i])).weekday() + 1 for i in range(len(final_dataframe['year']))]
    # dni i lata nie będą brane pod uwagę w predykcji bo konkretne dni dałyby overfit a lata się nie powtarzają więc predykcja nie ma sensu:    
        final_dataframe.drop(['day', 'year'], axis='columns', inplace=True)
    # dummy variables do regresji logistycznej:
        final_dataframe = pd.get_dummies(final_dataframe, columns=['time_of_day', 'weekday', 'month'])
        for i in range(1, 6):
            for j in range(6, 13):
                final_dataframe[final_dataframe.columns[j] + '*' + final_dataframe.columns[i]] = final_dataframe[final_dataframe.columns[i]] * final_dataframe[final_dataframe.columns[j]]
    # uwaga, interakcje zdefiniowane na podstawie kolejności nazw kolumn, zmiany w tworzeniu dummy variables mogą popsuć tą kolejność
    
    # estymacja modelu:
        model = LogisticRegression()
        model.fit(final_dataframe.drop('accidents', axis='columns'), final_dataframe['accidents'])
    # aktualna data i czas:
        now = datetime.datetime.now()
        time_of_day = '0-6' if now.hour in range(0, 6) else '6-9' if now.hour in range(6, 9) else '9-14' if now.hour in range(9, 14) else '14-18' if now.hour in range(14, 18) else '18-24'
        pred_dataframe = pd.DataFrame(columns=final_dataframe.columns[1:])
    
    # jednowierszowy dataframe z aktualnymi wartościami zmiennych niezależnych:
        for i in pred_dataframe.columns:
            pred_dataframe[i] = [0]
        pred_dataframe[f'time_of_day_{time_of_day}'] = 1
        pred_dataframe[f'weekday_{now.weekday() + 1}'] = 1
        pred_dataframe[f'month_{now.month}'] = 1
        pred_dataframe[f'weekday_{now.weekday() + 1}*time_of_day_{time_of_day}'] = 1

    # policzone prawdopodobieństwo:
        prob_val = round(pd.DataFrame(model.predict_proba(pred_dataframe)).loc[0, 1], 2)
        st.markdown(
            f"<p style='color: black; font-weight: bold; font-size: 1.1em;'>"
            f"Prawdopodobieństwo wypadku w godzinach {time_of_day}: {prob_val}%"
            f"</p>", 
            unsafe_allow_html=True
        )

else:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2: st.image("pl.gif", width='stretch')
