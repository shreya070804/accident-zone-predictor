import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import plotly.express as px
from datetime import datetime
import openrouteservice
from openrouteservice import convert
import base64
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import streamlit as st
import streamlit.components.v1 as components



st.markdown("## 📍 Get Your Current Location")

# Placeholders
lat_placeholder = st.empty()
lon_placeholder = st.empty()

# Add JS to fetch location
components.html("""
    <script>
        navigator.geolocation.getCurrentPosition(
            function(pos) {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                const input = window.parent.document.querySelectorAll('input[data-baseweb="input"]');
                input[0].value = lat.toFixed(6);
                input[1].value = lon.toFixed(6);
                input[0].dispatchEvent(new Event('input', { bubbles: true }));
                input[1].dispatchEvent(new Event('input', { bubbles: true }));
            },
            function(err) {
                alert("Location access denied or not available.");
            }
        );
    </script>
""", height=0)

# Manual fallback
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", key="lat", format="%.6f")
with col2:
    lon = st.number_input("Longitude", key="lon", format="%.6f")


# Set wide layout for mobile responsiveness
st.set_page_config(
    page_title="🚦 Accident Zone Predictor",
    layout="wide",  # Use wide layout for better mobile experience
    page_icon="🚦",
)
st.markdown("""
    <style>
        /* Make app responsive on mobile */
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        iframe {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)


# --- Theme Toggle ---
mode = st.sidebar.radio("🌗 Choose Theme", ["Dark", "Light"], key="theme")
if mode == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: white; }
            .stApp { background-color: #1e1e1e; color: white; }
        </style>
    """, unsafe_allow_html=True)

st.title("🚦 Accident-Prone Zone Predictor")
st.markdown("Enter your location below to see nearby accident data.")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("📍 Enter your Latitude", value=19.0760, format="%.6f")
with col2:
    lon = st.number_input("📍 Enter your Longitude", value=72.8777, format="%.6f")


# --- Sample Data ---
data = {
    'latitude': [lat + 0.001, lat - 0.0012, lat + 0.0007, lat - 0.0005, lat + 0.0015],
    'longitude': [lon + 0.002, lon - 0.0018, lon + 0.001, lon - 0.002, lon + 0.0012],
    'severity': [3, 2, 4, 5, 1],
    'date': [
        datetime(2025, 7, 10),
        datetime(2025, 7, 15),
        datetime(2025, 7, 18),
        datetime(2025, 7, 19),
        datetime(2025, 7, 5)
    ]
}
df = pd.DataFrame(data)

# --- Filter by Severity ---
st.sidebar.markdown("### 📊 Filter by Severity")
min_sev = int(df['severity'].min())
max_sev = int(df['severity'].max())
selected_severity_range = st.sidebar.slider("Select Severity Range", min_value=min_sev, max_value=max_sev, value=(min_sev, max_sev))
df = df[(df['severity'] >= selected_severity_range[0]) & (df['severity'] <= selected_severity_range[1])]

# --- Clustering ---
kmeans = KMeans(n_clusters=2, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# --- Filters ---
st.sidebar.header("⚙️ Filters")
selected_severity = st.sidebar.multiselect("Select Severity Levels", options=sorted(df["severity"].unique()), default=sorted(df["severity"].unique()))
df = df[df["severity"].isin(selected_severity)]


# --- Static Heatmap with Cluster Risk Markers ---
st.subheader("🌍 Static Accident Heatmap with Cluster Risk Markers")
m = folium.Map(location=[lat, lon], zoom_start=14)
HeatMap(data=df[['latitude', 'longitude', 'severity']], radius=15).add_to(m)

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    center_lat = kmeans.cluster_centers_[cluster_id][0]
    center_lon = kmeans.cluster_centers_[cluster_id][1]
    avg_severity = cluster_data['severity'].mean()

    if avg_severity >= 4:
        color = 'red'
        risk = "High Risk 🚨"
    elif avg_severity >= 2.5:
        color = 'orange'
        risk = "Medium Risk ⚠️"
    else:
        color = 'green'
        risk = "Low Risk ✅"

    folium.CircleMarker(
        location=(center_lat, center_lon),
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(f"Cluster {cluster_id}: {risk}<br>Avg Severity: {avg_severity:.2f}", max_width=300)
    ).add_to(m)

folium_static(m)

# --- Violin Chart ---
st.subheader("📊 Severity Distribution by Cluster")
violin_fig = px.violin(df, x="Cluster", y="severity", color="Cluster",
                       box=True, points="all",
                       color_discrete_sequence=px.colors.sequential.Viridis,
                       title="Severity Distribution Across Clusters")
st.plotly_chart(violin_fig, use_container_width=True)

# --- Top Dangerous Clusters ---
st.subheader("📍 Top Dangerous Zones by Cluster")
top_clusters = df.groupby('Cluster')['severity'].mean().sort_values(ascending=False).head(3)
for cluster_id in top_clusters.index:
    center = kmeans.cluster_centers_[cluster_id]
    st.markdown(f"**Cluster {cluster_id}**: Lat = {center[0]:.6f}, Lon = {center[1]:.6f}, Avg Severity = {top_clusters[cluster_id]:.2f}")

# --- Risk Level Badges ---
st.subheader("🎯 Cluster Risk Level Badges")
severity_badge = lambda sev: (
    "🚨 High" if sev >= 4 else 
    "⚠️ Medium" if sev >= 2.5 else 
    "✅ Low"
)

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_severity = cluster_data['severity'].mean()
    center = kmeans.cluster_centers_[cluster_id]

    st.markdown(f"""
        **Cluster {cluster_id}**  
        ➔ Center: ({center[0]:.6f}, {center[1]:.6f})  
        ➔ Avg Severity: {avg_severity:.2f}  
        ➔ Risk Level: {severity_badge(avg_severity)}
    """)
# --- AI-Based Cluster Captioning ---
st.subheader("🧠 AI-Based Danger Zone Captioning")
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def generate_caption(cluster_id, cluster_df):
    count = len(cluster_df)
    avg_sev = cluster_df['severity'].mean()

    # Determine peak time of day
    times = cluster_df['date'].dt.hour.apply(get_time_of_day)
    peak_time = times.mode()[0] if not times.empty else "Unknown"

    if avg_sev >= 4:
        severity_desc = "very high"
        emoji = "🚨"
    elif avg_sev >= 2.5:
        severity_desc = "moderate"
        emoji = "⚠️"
    else:
        severity_desc = "low"
        emoji = "✅"

    return (
        f"{emoji} **Cluster {cluster_id}** shows frequent **{peak_time.lower()}** accidents "
        f"with **{severity_desc} average severity** and **{count} accident(s)** reported."
    )

# Generate captions for each cluster
for cluster_id in sorted(df['Cluster'].unique()):
    caption = generate_caption(cluster_id, df[df['Cluster'] == cluster_id])
    st.markdown(f"- {caption}")


# --- Bar Chart ---
st.subheader("📊 Number of Accidents per Cluster")
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'count']
bar_fig = px.bar(cluster_counts,
                 x='Cluster', y='count',
                 labels={'Cluster': 'Cluster', 'count': 'Number of Accidents'},
                 color='Cluster')
bar_fig.update_layout(title="Accidents per Cluster")
st.plotly_chart(bar_fig, use_container_width=True)

# --- Risk Alert ---
st.subheader("🛡️ Risk Alert Near Your Location")
df['dist'] = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
nearby = df[df['dist'] < 0.002]
if not nearby.empty:
    avg_sev = nearby['severity'].mean()
    st.markdown(f"**🟠 Avg Severity Nearby:** {avg_sev:.2f}")
    if avg_sev >= 4:
        st.error("🚨 High-Risk Zone: Extreme accident severity nearby!")
    elif avg_sev >= 2.5:
        st.warning("⚠️ Medium Risk: Stay Alert in this area.")
    else:
        st.success("✅ Low Risk: Area appears relatively safe.")
else:
    st.info("ℹ️ No recent accidents found in this area.")

# --- Trends Over Time ---
st.subheader("📈 Accident Trends Over Time")
start_date = st.date_input("Start Date", value=df['date'].min().date())
end_date = st.date_input("End Date", value=df['date'].max().date())
filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
trend = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index(name='count')
trend_fig = px.line(trend, x='date', y='count', markers=True, title="📆 Daily Accident Trend")
trend_fig.update_traces(line=dict(width=4), marker=dict(size=8))
st.plotly_chart(trend_fig, use_container_width=True)

# --- Route Risk Analyzer ---
st.subheader("🛽 Route Risk Analyzer")
start = st.text_input("Enter Start Location")
end = st.text_input("Enter End Location")
ORS_API_KEY = "your_openrouteservice_api_key"

if st.button("Analyze Route") and start and end:
    try:
        client = openrouteservice.Client(key=ORS_API_KEY)
        coords = [
            client.pelias_search(start)['features'][0]['geometry']['coordinates'],
            client.pelias_search(end)['features'][0]['geometry']['coordinates']
        ]
        route = client.directions(coords)
        geometry = route['routes'][0]['geometry']
        decoded = convert.decode_polyline(geometry)

        route_map = folium.Map(location=[decoded['coordinates'][0][1], decoded['coordinates'][0][0]], zoom_start=13)
        folium.PolyLine(
            locations=[(coord[1], coord[0]) for coord in decoded['coordinates']],
            color="blue",
            weight=5,
            opacity=0.7
        ).add_to(route_map)
        folium.Marker(location=[coords[0][1], coords[0][0]], popup="Start", icon=folium.Icon(color='green')).add_to(route_map)
        folium.Marker(location=[coords[1][1], coords[1][0]], popup="End", icon=folium.Icon(color='red')).add_to(route_map)

        st.success("🚣️ Route successfully analyzed!")
        folium_static(route_map)
    except Exception as e:
        st.error(f"❌ Error fetching route: {e}")

# --- Download Summary ---
st.subheader("📅 Download Summary")
summary = f"Accident Summary from {start_date} to {end_date}\nTotal Accidents: {len(filtered_df)}"
b64 = base64.b64encode(summary.encode()).decode()
href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">📄 Download Summary</a>'
st.markdown(href, unsafe_allow_html=True)

# --- Severity Prediction ---
st.subheader("🔮 Predicted Accident Severity at Your Location")
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(df[['latitude', 'longitude']], df['severity'])
predicted_severity = knn.predict([[lat, lon]])[0]

badge = "🚨 High" if predicted_severity >= 4 else \
        "⚠️ Medium" if predicted_severity >= 2.5 else \
        "✅ Low"

st.markdown(f"""
    **📍 Location:** ({lat:.6f}, {lon:.6f})  
    **🤖 Predicted Severity:** {predicted_severity:.2f}  
    **🚩 Risk Level:** {badge}
""")

# --- Accidents by Time of Day ---
st.subheader("🕓 Accidents by Time of Day")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['date'].dt.hour.apply(get_time_of_day)
time_counts = df['TimeOfDay'].value_counts().reindex(['Morning', 'Afternoon', 'Evening', 'Night'])

time_fig = px.bar(
    x=time_counts.index,
    y=time_counts.values,
    labels={'x': 'Time of Day', 'y': 'Number of Accidents'},
    title="Accident Frequency by Time of Day",
    color=time_counts.index,
    color_discrete_sequence=px.colors.sequential.Redor
)
st.plotly_chart(time_fig, use_container_width=True)
# --- 🚘 Smart Traffic Advisory by Cluster ---
st.subheader("🚘 Smart Traffic Advisory by Cluster")

advice_dict = {
    "Morning": "🚶 Ideal time to commute with lower accident risks.",
    "Afternoon": "☀️ Stay alert — moderate activity and mixed severity.",
    "Evening": "🌇 Drive with caution — visibility and traffic congestion increase risk.",
    "Night": "🌙 High alert! Limited visibility and fatigue can raise accident chances."
}

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_df = df[df['Cluster'] == cluster_id]
    times = cluster_df['date'].dt.hour.apply(get_time_of_day)
    peak_time = times.mode()[0] if not times.empty else "Unknown"
    avg_sev = cluster_df['severity'].mean()

    st.markdown(f"""
    ### 🧭 Cluster {cluster_id} Travel Advisory
    - 🕓 Peak Accident Time: **{peak_time}**
    - 💥 Average Severity: **{avg_sev:.2f}**
    - 📢 Advisory: {advice_dict.get(peak_time, "🛑 Be cautious at all times.")}
    """)

import plotly.express as px

# --- Best Styled Accident Density Heatmap ---
st.subheader("🔥 Accident Severity Density Heatmap")

fig = px.density_heatmap(
    df,
    x="longitude",
    y="latitude",
    z="severity",  # Optional: highlights high severity zones
    nbinsx=60,
    nbinsy=60,
    color_continuous_scale="YlOrRd",  # Smooth gradient: Yellow → Orange → Red
    title="High-Density Accident Zones with Severity"
)

# Add layout styling
fig.update_layout(
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    title_x=0.5,
    title_font_size=20,
    margin=dict(l=10, r=10, t=60, b=10),
    height=550,
    plot_bgcolor='rgba(0,0,0,0)',
)

# Streamlit display
st.plotly_chart(fig, use_container_width=True)

# --- 🚨 Top 3 Most Dangerous Clusters ---
st.subheader("🚨 Top 3 Dangerous Zones")

if 'Cluster' in df.columns and 'severity' in df.columns:
    top_clusters = df.groupby('Cluster')['severity'].mean().sort_values(ascending=False).head(3)

    for idx, (cluster_id, avg_sev) in enumerate(top_clusters.items(), 1):
        st.markdown(f"### {idx}. Cluster {cluster_id} – Avg Severity: {avg_sev:.2f} 🚩")

        cluster_data = df[df['Cluster'] == cluster_id]
        center_lat = cluster_data['latitude'].mean()
        center_lon = cluster_data['longitude'].mean()

        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Cluster {cluster_id} - Danger Score: {avg_sev:.2f}",
            icon=folium.Icon(
                color="red" if idx == 1 else "orange" if idx == 2 else "darkred",
                icon="exclamation-triangle",
                prefix='fa'
            )
        ).add_to(m)

# --- 🌐 Display Final Map ---
folium_static(m)