import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Analyse de forages miniers",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {text}</a>'
    return href

# Fonction pour créer une représentation 3D des forages (version simplifiée)
def create_drillhole_3d_plot(collars_df, survey_df, 
                            hole_id_col, x_col, y_col, z_col,
                            azimuth_col, dip_col, depth_col):
    
    if collars_df is None or survey_df is None:
        return None
    
    fig = go.Figure()
    
    # Pour chaque trou de forage
    for hole_id in collars_df[hole_id_col].unique():
        # Récupérer les données de collar
        collar = collars_df[collars_df[hole_id_col] == hole_id]
        if collar.empty:
            continue
            
        # Point de départ du trou
        x_start = collar[x_col].values[0]
        y_start = collar[y_col].values[0]
        z_start = collar[z_col].values[0]
        
        # Récupérer les données de survey
        hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        if hole_surveys.empty:
            continue
            
        # Calculer les points 3D pour le tracé du trou
        x_points = [x_start]
        y_points = [y_start]
        z_points = [z_start]
        
        current_x, current_y, current_z = x_start, y_start, z_start
        prev_depth = 0
        
        for _, survey in hole_surveys.iterrows():
            depth = survey[depth_col]
            azimuth = survey[azimuth_col]
            dip = survey[dip_col]
            
            segment_length = depth - prev_depth
            
            # Convertir l'azimuth et le dip en direction 3D
            azimuth_rad = np.radians(azimuth)
            dip_rad = np.radians(dip)
            
            # Calculer la nouvelle position
            dx = segment_length * np.sin(dip_rad) * np.sin(azimuth_rad)
            dy = segment_length * np.sin(dip_rad) * np.cos(azimuth_rad)
            dz = -segment_length * np.cos(dip_rad)  # Z est négatif pour la profondeur
            
            current_x += dx
            current_y += dy
            current_z += dz
            
            x_points.append(current_x)
            y_points.append(current_y)
            z_points.append(current_z)
            
            prev_depth = depth
        
        # Ajouter la trace du trou de forage
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='lines',
                name=f'Forage {hole_id}',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                hovertext=[f'ID: {hole_id}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
                           for x, y, z in zip(x_points, y_points, z_points)]
            )
        )
    
    # Ajuster la mise en page
    fig.update_layout(
        title="Visualisation 3D des forages",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Élévation)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Titre de l'application
st.title('Analyse de données de forages miniers')

# Barre latérale pour la navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Sélectionnez une page:', [
    'Chargement des données', 
    'Aperçu des données', 
    'Visualisation 3D'
])

# Initialisation des variables de session
if 'collars_df' not in st.session_state:
    st.session_state.collars_df = None
    
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'hole_id': None,
        'x': None,
        'y': None,
        'z': None,
        'azimuth': None,
        'dip': None,
        'depth': None
    }

# Page de chargement des données
if page == 'Chargement des données':
    st.header('Chargement des données')
    
    # Créer des onglets pour les différents types de données
    tabs = st.tabs(["Collars", "Survey"])
    
    # Onglet Collars
    with tabs[0]:
        st.subheader('Chargement des données de collars')
        
        collars_file = st.file_uploader("Télécharger le fichier CSV des collars", type=['csv'])
        if collars_file is not None:
            st.session_state.collars_df = pd.read_csv(collars_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.collars_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.collars_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou", 
                                                                          [''] + cols, 
                                                                          index=0)
            with col2:
                st.session_state.column_mapping['x'] = st.selectbox("Colonne X", 
                                                                    [''] + cols,
                                                                    index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['y'] = st.selectbox("Colonne Y", 
                                                                    [''] + cols,
                                                                    index=0)
            with col2:
                st.session_state.column_mapping['z'] = st.selectbox("Colonne Z", 
                                                                    [''] + cols,
                                                                    index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données"):
                st.dataframe(st.session_state.collars_df.head())
    
    # Onglet Survey
    with tabs[1]:
        st.subheader('Chargement des données de survey')
        
        survey_file = st.file_uploader("Télécharger le fichier CSV des surveys", type=['csv'])
        if survey_file is not None:
            st.session_state.survey_df = pd.read_csv(survey_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.survey_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.survey_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Survey)", 
                                                                          [''] + cols, 
                                                                          index=0)
            with col2:
                st.session_state.column_mapping['depth'] = st.selectbox("Colonne profondeur", 
                                                                        [''] + cols,
                                                                        index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['azimuth'] = st.selectbox("Colonne azimut", 
                                                                          [''] + cols,
                                                                          index=0)
            with col2:
                st.session_state.column_mapping['dip'] = st.selectbox("Colonne pendage", 
                                                                      [''] + cols,
                                                                      index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Survey)"):
                st.dataframe(st.session_state.survey_df.head())

# Page d'aperçu des données
elif page == 'Aperçu des données':
    st.header('Aperçu des données')
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger des données.")
    else:
        # Créer des onglets pour les différents types de données
        data_tabs = st.tabs(["Collars", "Survey"])
        
        # Onglet Collars
        with data_tabs[0]:
            if st.session_state.collars_df is not None:
                st.subheader('Données de collars')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.collars_df)}")
                st.dataframe(st.session_state.collars_df)
                
                st.markdown(get_csv_download_link(st.session_state.collars_df, "collars_data.csv", "les données de collars"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de collars n'a été chargée.")
        
        # Onglet Survey
        with data_tabs[1]:
            if st.session_state.survey_df is not None:
                st.subheader('Données de survey')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.survey_df)}")
                st.dataframe(st.session_state.survey_df)
                
                st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data.csv", "les données de survey"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de survey n'a été chargée.")

# Page de visualisation 3D
elif page == 'Visualisation 3D':
    st.header('Visualisation 3D des forages')
    
    # Vérifier si les données nécessaires ont été chargées
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.warning("Les données de collars et de survey sont nécessaires pour la visualisation 3D. Veuillez les charger d'abord.")
    else:
        # Vérifier si les colonnes nécessaires ont été spécifiées
        required_cols = ['hole_id', 'x', 'y', 'z', 'azimuth', 'dip', 'depth']
        missing_cols = [col for col in required_cols if st.session_state.column_mapping[col] is None or st.session_state.column_mapping[col] == '']
        
        if missing_cols:
            st.warning(f"Certaines colonnes requises n'ont pas été spécifiées: {', '.join(missing_cols)}. Veuillez les définir dans l'onglet 'Chargement des données'.")
        else:
            # Options pour la visualisation
            st.subheader("Options de visualisation")
            
            # Sélection des forages à afficher
            hole_id_col = st.session_state.column_mapping['hole_id']
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            selected_holes = st.multiselect("Sélectionner les forages à afficher", all_holes, default=all_holes[:min(5, len(all_holes))])
            
            # Filtrer les données selon les forages sélectionnés
            if selected_holes:
                filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(selected_holes)]
                filtered_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(selected_holes)]
                
                # Créer la visualisation 3D
                fig_3d = create_drillhole_3d_plot(
                    filtered_collars, filtered_survey,
                    hole_id_col=hole_id_col,
                    x_col=st.session_state.column_mapping['x'],
                    y_col=st.session_state.column_mapping['y'],
                    z_col=st.session_state.column_mapping['z'],
                    azimuth_col=st.session_state.column_mapping['azimuth'],
                    dip_col=st.session_state.column_mapping['dip'],
                    depth_col=st.session_state.column_mapping['depth']
                )
                
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.error("Impossible de créer la visualisation 3D avec les données fournies.")
            else:
                st.info("Veuillez sélectionner au moins un forage à afficher.")