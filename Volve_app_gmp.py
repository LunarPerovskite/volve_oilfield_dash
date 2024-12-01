#GeoFieldvisor Beta release _ created by JEMR


import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx, dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_leaflet as dl
import plotly.graph_objects as go
import base64
from scipy.signal import savgol_filter  # Para suavizado de datos


resolution_3d=0.1
opaciti_3d=1
samples_logs=20

"LOGO"

logo_filename =  os.path.join("assets", "Logo.png")
encoded_logo = base64.b64encode(open(logo_filename, 'rb').read()).decode() 

programlogo_filename =  os.path.join("assets", "FieldVisor_iconpng.png")
encoded_programlogo = base64.b64encode(open(programlogo_filename, 'rb').read()).decode() 

"Apartado datos tab 3D"
# Ruta al archivo
formations_path = os.path.join("Data", "Surfaces", "Formations")
# Ruta al directorio de pozos
wells_path = os.path.join("data", "Wells", "Path")
# Ruta al directorio de contactos
contacs_path = os.path.join("data", "Surfaces", "Contacts")
# Rutas predefinidas a los archivos TIFF
tif_oil_top = os.path.join("data", "Surfaces", "Contacts", "oil_top.tif")
tif_water_contact = os.path.join("data", "Surfaces", "Contacts", "water_contact.tif")
#Rutas fallas
faults_path = os.path.join("data", "Faults")
# Función para leer y reducir la resolución de los datos TIFF
def get_tif_coordinates(tif_path, reduction_factor=resolution_3d):
    with rasterio.open(tif_path) as src:
        # Leer la primera banda del archivo TIFF (suponiendo que solo haya una banda de datos)
        tif_data = src.read(1)
        
        # Obtener la transformada del raster (información sobre el desplazamiento y escala de las coordenadas)
        transform = src.transform
        
        # Obtener las dimensiones del raster
        height, width = tif_data.shape

        # Generar malla de índices
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Convertir las coordenadas de píxeles a coordenadas geográficas usando la transformada
        x_coords, y_coords = rasterio.transform.xy(transform, y.flatten(), x.flatten())
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        # Los valores de z corresponden a los valores de la primera banda del raster
        z_coords = tif_data.flatten()

        # Reducir la cantidad de datos para mejorar el rendimiento (muestreo aleatorio)
        # Reducir los datos según el factor de reducción (por ejemplo, 5% de los puntos)
        num_samples = int(len(x_coords) * reduction_factor)
        sample_indices = np.random.choice(len(x_coords), size=num_samples, replace=False)

        # Reducir las coordenadas y valores de z
        x_coords_reduced = x_coords[sample_indices]
        y_coords_reduced = y_coords[sample_indices]
        z_coords_reduced = z_coords[sample_indices]
        
        return x_coords_reduced, y_coords_reduced, z_coords_reduced

# Usar la función para obtener las coordenadas del archivo "oil_top"
x_oil_top, y_oil_top, z_oil_top = get_tif_coordinates(tif_oil_top)

# Usar la función para obtener las coordenadas del archivo "water_contact"
x_water_contact, y_water_contact, z_water_contact = get_tif_coordinates(tif_water_contact)

# Obtener la lista de archivos disponibles en el directorio, ignorando archivos .xml
available_surfaces = [
    file for file in os.listdir(formations_path)
    if (file.endswith("") or file.endswith(".dat")) and not file.endswith(".xml")  # Excluye .xml
]

# Obtener la lista de archivos y extraer los nombres de los pozos
available_wells = [file.replace(".txt", "") for file in os.listdir(wells_path) if file.endswith("")]





"Apartado datos Tab de Pozo"

# Ruta de la carpeta con archivos LAS
logs_folder = os.path.join("data", "Wells", "Logs")
# Ruta de los archivos de tope
tops_folder = os.path.join("data", "Wells", "Tops")

# Leer archivos LAS disponibles
def get_las_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith('')]

# Función para leer el archivo LAS y procesar sus curvas
def process_las_file(file_path, columns_to_load=None, sample_interval=samples_logs):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    start_idx = None
    columns = []

    # Procesar solo hasta ~Ascii, excluyendo ~Parameter
    for i, line in enumerate(lines):
        if "~Ascii" in line:
            start_idx = i + 1
            break
        elif "~Curve" in line:  # Extraer nombres desde el bloque ~Curve
            while not lines[i + 1].startswith("#"):
                i += 1
                curve_line = lines[i].strip()
                if curve_line:
                    curve_name = curve_line.split()[0]
                    columns.append(curve_name)

    # Asegurarse de que no incluimos ~Parameter ni DEPTH/DEPT
    if "~Parameter" in lines:
        lines = [line for line in lines if "~Parameter" not in line]

    if start_idx is None or not columns:
        raise ValueError(f"El archivo {file_path} no tiene datos válidos.")

    # Leer los datos usando las columnas detectadas
    data = pd.read_csv(
        file_path,
        skiprows=start_idx,
        sep='\s+',
        names=columns,
        na_values=-999.25  # Manejar valores nulos
    )

    # Tomar una muestra de los datos (si se especifica)
    data_sampled = data.iloc[::sample_interval, :]

    return data_sampled, columns

# Función para eliminar puntos anómalos utilizando IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

# Función para aplicar un suavizado a los datos
def smooth_data(df, column):
    # Crear una copia explícita del DataFrame para evitar la advertencia
    df = df.copy()
    # Suavizado simple usando el filtro Savitzky-Golay
    smoothed = savgol_filter(df[column], window_length=11, polyorder=2)
    df[column + '_smoothed'] = smoothed
    return df

# Función para leer los archivos de tope
def read_top_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Saltar las líneas de comentarios e información del encabezado
    header_start = lines.index("BEGIN HEADER\n")
    header_end = lines.index("END HEADER\n")

    # Extraer las líneas después del encabezado
    data_lines = lines[header_end + 1:]

    # Procesar las líneas de datos
    valid_data = []
    for line in data_lines:
        # Saltar líneas vacías
        if not line.strip():
            continue
        
        # Dividir la línea y extraer columnas necesarias
        columns = line.split()
        if len(columns) >= 4:  # Verificar que tiene al menos 4 columnas
            try:
                md = float(columns[0])  # Primera columna: MD
                unit = columns[3].strip('"')  # Cuarta columna: Surface name
                valid_data.append([md, unit])
            except ValueError:
                # Manejar casos donde la conversión falle (p. ej., MD no es un número)
                continue

    # Convertir los datos válidos en un DataFrame
    df = pd.DataFrame(valid_data, columns=["MD", "Unit"])
    return df

# Texto con las descripciones de los logs
log_descriptions = """
DEPT: Depth in meters (m).
DEPTH: Depth in meters (m).
CALI: Caliper log, measures borehole diameter (inches).
CARB_FLAG: Carbonate presence indicator.
COAL_FLAG: Coal presence indicator.
DT: Sonic transit time (microseconds/ft).
DRHO: Density correction, indicates mudcake thickness (g/cm³).
DRHU: Hydrocarbon density correction (kg/m³).
GR: Gamma Ray, measures natural radioactivity (gAPI).
NPHI: Neutron porosity (m³/m³).
PEF: Photoelectric factor, related to lithology (b/e).
RD: Deep resistivity, uncorrected (ohm·m).
RHOB: Formation density (g/cm³).
ROBB: Bulk density corrected for borehole effects (g/cm³).
ROBU: Uncorrected bulk density (g/cm³).
ROP5_RM: Average rate of penetration (unknown units).
RS: Relative saturation (sm³/sm³).
RT: True formation resistivity (ohm·m).
RDEP: Deep resistivity (ohm·m).
RMED: Medium resistivity (ohm·m).
ROP: Rate of penetration (m).
SW: Water saturation.
VSH: Shale volume fraction.
BVW: Bound water volume.
KLOGH: Permeability-related parameter.
PHIF: Effective porosity (m³/m³).
SAND_FLAG: Sand presence indicator.
N/G: Net-to-gross ratio.
"""



"APARTADO INICIO"

# Ruta del archivo CSV de producción
file_path_production = os.path.join("data", "Production", "Production_year.csv")
# Ruta del archivo shapefile
shp_path = os.path.join("data", "Shp_field", "Volve_form.shp")
# Leer shapefile con geopandas
shapefile = gpd.read_file(shp_path)

# Convertir geometrías a GeoJSON
geojson_data = shapefile.__geo_interface__

def pie_chart_rec():
    # Datos para el gráfico de pastel
    labels = ['oil [mill Sm3]', 
              'gas [bill Sm3]',
              'NGL [mill tonn]',
              'cond. [mill Sm3]',]
    
    colors = ['green', 'red',"yellow","orange"]
    values = [10.17, 0.81, 0.16, 0.09]
        
   

    # Crear el gráfico de pastel
    return go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.3,marker=dict(colors=colors))],
        layout={
            'title': 'Orig. Recoverable',  # Título dentro del gráfico
            'title_x': 0.5,  # Centrar el título

            },
        )

# Función que genera el segundo gráfico de pastel
def pie_chart_plc():
    # Datos para el segundo gráfico de pastel
    labels_2 = ['oil [mill Sm3]',            
                'gas [bill Sm3]',
                ]
    # Colores para los segmentos
    colors = ['blue', 'violet']
    values_2 = [19.00, 3.00, ]

    # Crear el segundo gráfico de pastel
    return go.Figure(
        data=[go.Pie(labels=labels_2, values=values_2, hole=0.3,marker=dict(colors=colors))],
        layout={
            'title': 'Orig. Inplace',  # Título dentro del gráfico
            'title_x': 0.5,  # Centrar el título

            },
        )

def get_field_info_data():
    return [
        {"Type": "Development", "Text": "Volve is a field in the central part of the North Sea, five kilometres north of the Sleipner Øst field. The water depth is 80 metres. Volve was discovered in 1993, and the plan for development and operation (PDO) was approved in 2005. The field was developed with a jack-up processing and drilling facility. The vessel 'Navion Saga' was used for storing stabilised oil. The production started in 2008."},
        {"Type": "Reservoir", "Text": "Volve produced oil from sandstone of Middle Jurassic age in the Hugin Formation. The reservoir is at a depth of 2700-3100 metres. The western part of the structure is heavily faulted and communication across the faults is uncertain."},
        {"Type": "Recovery strategy", "Text": "The field was produced with water injection for pressure support."},
        {"Type": "Transport", "Text": "The oil was exported by tankers and the rich gas was transported to the Sleipner A facility for further export."},
        {"Type": "Status", "Text": "Volve was shut down in 2016, and the facility was removed in 2018."}
    ]

# Crear la función para generar el histograma apilado
def generate_stacked_histogram():
    # Leer los datos del archivo CSV
    df = pd.read_csv(file_path_production, sep=';')
    fig = go.Figure()

    # Añadir cada categoría como una barra apilada
    fig.add_trace(go.Bar(x=df['YEAR'], y=df['OIL'], name='OIL', marker_color='green'))
    fig.add_trace(go.Bar(x=df['YEAR'], y=df['Condensate'], name='Condensate', marker_color='orange'))
    fig.add_trace(go.Bar(x=df['YEAR'], y=df['NGL'], name='NGL', marker_color='yellow'))
    fig.add_trace(go.Bar(x=df['YEAR'], y=df['Gas'], name='Gas', marker_color='red'))

    # Actualizar el layout para apilar las barras
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        #title="Yearly Production",
        xaxis_title="Year",
        yaxis_title="Production (mill Sm3)",
        xaxis=dict(
            tickmode='linear',  # Asegura que todos los años estén visibles
            dtick=1  # Esto asegura que los ticks estén de uno en uno
        ),
        #template="plotly_dark"  # Estilo oscuro
    )

    return fig





# Inicialización de la variable de cámara
camera_initialized = False
#dark template for dropdown and tabs
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"



# Crear la app Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])
load_figure_template(["darkly"])
# Establecer el título de la aplicación
app.title = 'FieldVisor'  # Aquí puedes cambiar el nombre
app._favicon = os.path.join("FieldVisor.ico")  # Ruta al favicon


# Layout inicial de la aplicación
tabs = html.Div([
    
    # Tabs para organizar la vista
    dcc.Tabs([
        
        #Tab Generalidades del Campo - reservas , años produciendo , produccion anual , unidades informacion
        dcc.Tab(label='Field Info', children=[
            html.Div([
                # Agregar el mapa interactivo
                dl.Map([
                    dl.TileLayer(),  # Fondo del mapa
                    dl.GeoJSON(
                        data=geojson_data,  # Datos del shapefile en formato GeoJSON
                        id="geojson-layer",

                        
                        style={"color": "green", "weight": 2, "fillColor": "green", "fillOpacity": 0.5},  # Estilo verde
                        #zoomToBounds=True,  # Ajustar la vista al GeoJSON
                        hoverStyle=dict(weight=5, color="red"),  # Estilo al pasar el cursor
                        
                       
                    ),
                    # Agregar la barra de escala (ScaleControl)
                    dl.ScaleControl(position="bottomright", maxWidth=200, metric=True, imperial=False)
                   
                    
                ],
                style={
                    'color': 'black',
                    "width": "400px",
                    "display": "inline-block",
                    #"vertical-align": "top",
                    "height": "300px",
                    "backgroundColor": "#333333",
                    "borderRadius": "10px",
                    #"padding": "10px",
                    'border': '1px solid #555555',
                    "position": "absolute",
                    "left": "5%", 
                    "top": "25%",
                },  
                center=[58.4381, 1.8868],  # Centrar mapa
                zoom=13,  # Nivel de zoom inicial
                id="map",
                ),
                
                html.Div([
                    #html.H3("Orig. Recoverable Resources", style={"textAlign": "center", "color": "white"}),
                    dcc.Graph(
                        id='pie-chart-recov',
                        figure=pie_chart_rec(), # El gráfico de pastel interactivo
                        style={
                            'color': 'black',
                            "width": "400px",
                            "display": "inline-block",
                            #"vertical-align": "top",
                            "height": "300px",
                            "backgroundColor": "#333333",
                            "borderRadius": "10px",
                            #"padding": "10px",
                            'border': '1px solid #555555',
                            "position": "absolute",
                            "left": "65%", 
                            "top": "25%",
                        },
                        config={
                          'displayModeBar': False,  #quitar simbolos de plotly
                          'showTips': False,
                          'displaylogo': False
                          
                        },
                    )
                ]),
                html.Div([
                    #html.H3("Orig. Inplace Resources", style={"textAlign": "center", "color": "white"}),
                    dcc.Graph(
                        id='pie-chart-place',
                        figure=pie_chart_plc(),  # Llamamos a la función para obtener el segundo gráfico de pastel
                        style={
                            'color': 'black',
                            "width": "400px",
                            "display": "inline-block",
                            #"vertical-align": "top",
                            "height": "300px",
                            "backgroundColor": "#333333",
                            "borderRadius": "10px",
                            #"padding": "10px",
                            'border': '1px solid #555555',
                            "position": "absolute",
                            "left": "35%", 
                            "top": "25%",
                        },
                        config={
                          'displayModeBar': False,  #quitar simbolos de plotly
                          'showTips': False,
                          'displaylogo': False
                          
                        },
                    )
                ]),
                
                html.Div([
                    dash_table.DataTable(
                        id='field-info-table',
                        data=get_field_info_data(),  # Llamamos a la función para obtener los datos de la tabla
                        columns=[
                            {"name": "Type", "id": "Type"},
                            {"name": "Text", "id": "Text"},
                            
                        ],
                        style_table={
                            'width': '60%',  # Ancho de la tabla
                            'float': 'left',  # Para dejar espacio a la izquierda
                        },
                        style_cell_conditional=[
                            {
                                'if': {'column_id': 'Type'},
                                'whiteSpace': 'nowrap',  # Evitar que se ajuste el texto en la columna "Type"
                                'textAlign': 'left',
                                'color': 'white',
                                'backgroundColor': '#232629'
                            },
                            {
                                'if': {'column_id': 'Date updated'},
                                'whiteSpace': 'nowrap',  # Evitar que se ajuste el texto en la columna "Date updated"
                                'textAlign': 'left',
                                'color': 'white',
                                'backgroundColor': '#232629'
                            },
                            {
                                'if': {'column_id': 'Text'},
                                'whiteSpace': 'normal',  # Permitir el ajuste de texto en la columna "Text"
                                'textAlign': 'left',
                                'color': 'white',
                                'backgroundColor': '#232629'
                            },
                        ],
                        style_header={
                            'display': 'none',  # Quitar el encabezado
                        },
                        style_data={
                            'backgroundColor': '#232629',  # Color de fondo oscuro para las celdas
                            'color': 'white'
                        },
                        style_as_list_view=True,  # Estilo tipo lista (sin bordes gruesos)
                    )
                ],style={
                    #'color': 'black',
                    "width": "100%",
                    #"display": "inline-block",
                    #"vertical-align": "top",
                    #"height": "400px",
                    #"backgroundColor": "#333333",
                    #"borderRadius": "10px",
                    #"padding": "10px",
                    #'border': '1px solid #555555',
                    "position": "absolute",
                    "left": "36%", 
                    "top": "71%",
                   }),
                
                
                
                
                html.Div([
                    
                    dcc.Graph(
                        id='production-histogram',
                        figure=generate_stacked_histogram(),  # Llamamos a la función para generar el histograma
                        style={
                            'color': 'black',
                            "width": "400px",
                            #"display": "inline-block",
                            #"vertical-align": "top",
                            "height": "300px",
                            "backgroundColor": "#333333",
                            "borderRadius": "10px",
                            #"padding": "10px",
                            'border': '1px solid #555555',
                            "position": "absolute",
                            "left": "5%", 
                            "top": "75%",
                        }, 
                        config={
                          'displayModeBar': False,  #quitar simbolos de plotly
                          'showTips': False,
                          'displaylogo': False,
                          
                        }, 
                    )
                ]),
                
            ])
        ]),
        
        # Tab 3D View
        dcc.Tab(label='3D View', children=[
            # Dropdown para seleccionar la superficie
            html.Div([
                dcc.Dropdown(
                    id='surface-dropdown',
                    options=[{'label': surface, 'value': surface} for surface in available_surfaces],
                    value=str(''),  # Superficie seleccionada por defecto
                    multi=True,  # Permitir selección múltiple
                    #clearable=False,  # Evita limpiar el valor seleccionado
                    placeholder="Select Units",
                )
            ], 
                style={
                'color': 'black',
                "width": "40%",
                "display": "inline-block",
                "vertical-align": "top",
                "height": "60px",
                "backgroundColor": "#333333",
                "borderRadius": "10px",
                "padding": "10px",
                'border': '1px solid #555555',
                "position": "absolute",
                "left": "0%",  # Colocado a la izquierda del gráfico
                "top": "25%",
            }),

            # Dropdown para seleccionar los pozos
            html.Div([
                dcc.Dropdown(
                    id='wells-dropdown',
                    options=[{'label': well, 'value': well} for well in available_wells],
                    value=None,
                    multi=True,
                    clearable=True,
                    placeholder="Select Wells"
                )
            ], style={
                'color': 'black',
                "width": "40%",
                "display": "inline-block",
                "vertical-align": "top",
                "height": "60px",
                "backgroundColor": "#333333",
                "borderRadius": "10px",
                "padding": "10px",
                'border': '1px solid #555555',
                "position": "absolute",
                "left": "0%", 
                "top": "35%",
            }),

            # Botones y slider
            html.Div([
                # Botones Oil, Water y Faults
                html.Div([
                    html.Button('Oil', id='oil-top-button', n_clicks=0, style={
                        
                        
                        
                    }),
                    html.Button('Water', id='water-contact-button', n_clicks=0, style={
                        
                       
                        
                    }),
                    html.Button('Faults', id='faults-button', n_clicks=0, style={
                        
                       
                        
                    }),
                ], style={
                    #'backgroundColor': '#333333',
                    #'color': 'white',
                    #'borderRadius': '10px',
                    #'marginBottom': '10px',
                    #'fontSize': '16px',
                    #'border': '0px solid #555555',
                    #position": "absolute",  # Posicionarlos a la izquierda
                    #"marginTop": "15%",
                    #"marginLeft": "5%",
                    #'width': '350px',  # Más anchos
                    #'height': '50px',
                    #"top": "80px",  # Ajustar la posición desde la parte superior
                    #"left": "20%",  # Ajustar la posición desde la izquierda
                    #"display": "flex",
                    "flexDirection": "column",  # Alinear los botones verticalmente
                    #"alignItems": "flex-start",  # Alinearlos a la izquierda
                    #"zIndex": "1",  # Asegurarse de que estén por encima de otros componentes si es necesario
                    
                }),

                
            ]),
            # Slider 
                html.Div([
                    html.Label("Vertical Exaggeration", style={'color': 'white', "textAlign": "center", 'width': '100%', 'marginBottom': '10px'}),
                    dcc.Slider(
                        id='vertical-exaggeration-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=1,
                        marks={i: f"{i}x" for i in range(1, 6)},
                        #vertical=False,  # Slider vertical
                    ),
                ], style={
                    #"display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center",
                    "marginTop": "7%",
                    "marginLeft": "7%",  # Ajusta la posición del slider hacia la izquierda
                    "height": "40px",  # Altura del slider
                    "width": "300px",    # Ancho del slider
                   
                }),
            # Gráfico
            dcc.Graph(id='3d-surface-plot',
                      style={
                          "height": "500px",  # Altura del gráfico
                          "width": "59%",     # Ancho del gráfico
                          "position": "fixed",  # Fija el gráfico en la pantalla
                          "top": "22%",        # Ajusta la posición desde la parte superior
                          "left": "40%",       # Ajusta la posición desde la izquierda
                          "zIndex": "1",       # Asegura que el gráfico esté por encima de otros elementos
                          'border': '1px solid #555555',  # Borde del gráfico
                      },
                      config={
                          #'displayModeBar': False,  #quitar simbolos de plotly
                          'showTips': False,
                          'displaylogo': False
                          
                      }),

            # Almacenar la posición de la cámara
            dcc.Store(id='camera-position', data={'x': 0, 'y': -1, 'z': 2}),
            
            html.Div(
            "SRC ED50 UTM 31",
            style={
                "position": "fixed",  # Fijo en la pantalla, no cambia con el scroll
                "bottom": "10px",     # Distancia desde la parte inferior
                "left": "10px",       # Distancia desde la parte izquierda
                "fontSize": "12px",   # Tamaño pequeño del texto
                "color": "gray",     # Color del texto
                #"fontFamily": "Arial, sans-serif",  # Fuente del texto
                "backgroundColor": "transparent",  # Sin fondo
                "zIndex": "1000"      # Asegura que se muestre sobre otros elementos
            }
)
        ]),  # Fin del tab 3D View






        # Tab Well View
        dcc.Tab(label='Well View', children=[
            html.Div([
                html.Div([
                        dcc.Dropdown(
                        id='las-file-dropdown',
                        options=[{'label': f, 'value': f} for f in get_las_files(logs_folder)],
                        placeholder="Select a Well"
                    ),
                ],style={
                'color': 'black',
                "width": "20%",
                "display": "inline-block",
                "vertical-align": "top",
                "height": "60px",
                "backgroundColor": "#333333",
                "borderRadius": "10px",
                "padding": "10px",
                'border': '1px solid #555555',
                "position": "absolute",
                "left": "0%",  # Colocado a la izquierda del gráfico
                "top": "25%",
                }),

                html.Div([
                    dcc.Dropdown(
                        id='curve-dropdown',
                        placeholder="Select Log",
                        multi=True,
                        #max_selected_items=5,
                        #maxHeight=200
                    ),
                ], style={
                'color': 'black',
                "width": "20%",
                "display": "inline-block",
                "vertical-align": "top",
                "height": "90px",
                "backgroundColor": "#333333",
                "borderRadius": "10px",
                "padding": "10px",
                'border': '1px solid #555555',
                "position": "absolute",
                "left": "0%",  # Colocado a la izquierda del gráfico
                "top": "35%",
                }),
                
            html.Button("Unit Tops", id='toggle-lines-button', n_clicks=0,
                        style={
                'color': 'white',
                "width": "8%",
                "display": "inline-block",
                "vertical-align": "top",
                "height": "60px",
                "backgroundColor": "#333333",
                "borderRadius": "10px",
                "padding": "10px",
                'border': '1px solid #555555',
                "position": "absolute",
                "left": "5%",  # Colocado a la izquierda del gráfico
                "top": "55%",
                }),
            
            html.Div([
                dcc.Store(id="sidebar-state", data=False),  # Estado de la barra lateral
                html.Div(
                    id="sidebar",
                    children=[
                        html.Button("Close", id="toggle-button", className="toggle-btn"),
                        html.Div(
                            id="sidebar-content",
                            children=[html.P(line.strip()) for line in log_descriptions.strip().split("\n")],
                            style={"padding": "15px", "color": "white", "font-size": "14px", "line-height": "1.6","zIndex": "1",},
                        ),
                    ],
                    style={
                        "position": "fixed",
                        "top": "0",
                        "left": "0",
                        "height": "100%",
                        "width": "0",
                        "overflow-x": "hidden",
                        "transition": "0.3s",
                        "background-color": "#111",
                    },
                ),
                html.Button("Logs Names", id="open-button", className="open-btn",
                            style={"position": "fixed", "left": "0", "top": "10px", "z-index": "1000", 
                                   "color": 'black', 'border': '1px solid #555555', "padding": "10px", "cursor": "pointer"}),
                            ]),

                html.Div(id='log-graph-container')  # Contenedor para los gráficos,
            ]),
        ]),
        dcc.Tab(label='About', children=[
            html.Div([
                html.H3("About the Data and Licences", style={"textAlign": "center", "color": "white"}),
                html.P(
                    """
                    The beta version of the file set was released in June 2018 and replaced by a new version in October 2018. 
                    The files were subject to a licence based on a Creative Commons licence to ensure a standardised, 
                    internationally-recognised approach to licencing the data, and to protect the rights of the data owners. 
                    As of September 2020, the Equinor Open Data Licence (download below) supersedes the Creative Commons 
                    (CC BY-NC-SA 4.0) and now applies retrospectively to all data, meaning that they shall not be resold, 
                    and, if shared, the data owners shall be attributed. NB. Please note that this also applies 
                    retrospectively to all previously published files in the Volve dataset beta version.
                    """,
                    style={"textAlign": "justify", "color": "white", "padding": "10px"}
                ),
                html.P(
                    """
                    These data can be accessed at the following link: 
                    """,
                    style={"textAlign": "justify", "color": "white", "padding": "10px"}
                ),
                html.A(
                    "https://www.equinor.com/energy/volve-data-sharing",
                    href="https://www.equinor.com/energy/volve-data-sharing",
                    target="_blank",
                    style={"color": "#1E90FF", "textDecoration": "none"}
                ),
                html.P(
                    """
                    The production and reserves data were extracted from the following sources:
                    
                    - Norsk Petroleum
                    
                    - SODIR Facts
                    """,
                    style={"textAlign": "justify", "color": "white", "padding": "10px"}
                ),
                html.A(
                    "https://www.norskpetroleum.no/en/facts/field/volve/",
                    href="https://www.norskpetroleum.no/en/facts/field/volve/",
                    target="_blank",
                    style={"color": "#1E90FF", "textDecoration": "none"}
                ),
                html.Br(),  # Salto de línea para separar los enlaces
                html.A(
                    "https://factpages.sodir.no/en/field/pageview/all/3420717",
                    href="https://factpages.sodir.no/en/field/pageview/all/3420717",
                    target="_blank",
                    style={"color": "#1E90FF", "textDecoration": "none"}
                ),
                html.P(
                    """
                    This example demonstrates the capabilities of this program. However, it is not used for 
                    commercial purposes, nor to sell or distribute any related to this dataset.
                    """,
                    style={"textAlign": "justify", "color": "white", "padding": "10px"}
                )
            ], style={"backgroundColor": "#232629", "padding": "20px"})
        ]),
    
    ]),
        
        
],className="dbc")

# Layout principal con las Tabs
app.layout = dbc.Container(
    [
        html.A(
            html.Img(
            src='data:image/png;base64,{}'.format(encoded_programlogo),
            style={
                'height': '4em',  # Hacer la imagen más grande
                'width': 'auto',
                'display': 'block',
                'position': 'absolute',  # Asegura que la imagen no desplace el texto
                'top': '1%',
                'left': '30%',
                }
            ),
            
        ),
        html.A(
            html.Img(
            src='data:image/png;base64,{}'.format(encoded_logo),
            style={
                'height': '4em',  # Hacer la imagen más grande
                'width': 'auto',
                'display': 'block',
                'position': 'absolute',  # Asegura que la imagen no desplace el texto
                'top': '0%',
                'left': '2%',
                }
            ),
            href="https://www.gearsmap.com",  # Enlace
            target="_blank",  # Abrir en nueva pestaña
            style={'height': '2em', 'width': '2em',"textDecoration": "none",'display': 'block','position': 'absolute'}  # Estilo para quitar el subrayado del enlace
        ),
        html.H1(
            "Volve Oilfield Visor",
            style={"textAlign": "center", 'position': 'relative', 'top': '0%'},  # Alineación del texto arriba
            className="text-center mb-2"
        ),
        tabs,
    ],
    fluid=True,
    className="mt-2",
)

# Callback para actualizar el gráfico con la superficie seleccionada y la exageración vertical
@app.callback(
    Output('3d-surface-plot', 'figure'),
    [Input('surface-dropdown', 'value'),
     Input('vertical-exaggeration-slider', 'value'),
     Input('wells-dropdown', 'value'),
     Input('camera-position', 'data'),
     Input('oil-top-button', 'n_clicks'),
     Input('water-contact-button', 'n_clicks'),
     Input("faults-button", "n_clicks")]
)
def update_figure(selected_surfaces, vertical_exaggeration, selected_wells, camera_position, oil_top_clicks, water_contact_clicks, faults_button_clicks):
    global camera_initialized
    # Si no hay superficies seleccionadas, devolver una figura vacía
    #if not selected_surfaces:
        
    #   return go.Figure()  # Retorna una figura vacía si no hay selección
    # Crear la figura 3D


    fig = go.Figure()
    
    # Iterar sobre las superficies seleccionadas
    for selected_surface in selected_surfaces:
        
        # Ruta al archivo seleccionado
        file_path = os.path.join(formations_path, selected_surface)
        
        # Leer el archivo y cargar los datos
        column_names = ["x", "y", "z", "column", "row"]
        data_start_line = 20  # Línea donde comienzan los datos
        data = pd.read_csv(file_path, skiprows=data_start_line, sep='\s+', names=column_names)

        # Convertir z a valores negativos (profundidad)
        data["z"] = -data["z"]

        # Reducir la calidad: seleccionar una muestra de puntos
        data_reduced = data.sample(frac=resolution_3d, random_state=42)
        # Escalar z según la exageración vertical
        scaled_z = data_reduced["z"] 
        # Añadir cada superficie como un trazo
        fig.add_trace(go.Mesh3d(
            x=data_reduced["x"],
            y=data_reduced["y"],
            z=scaled_z,
            intensity=data_reduced["z"],  # Basado en la profundidad original
            colorscale='Jet',  # Usar colormap Jet
            opacity=opaciti_3d,
            showscale=False,
            name=selected_surface,  # Nombre de la superficie en la leyenda
            hoverinfo='x+y+z',  # Información a mostrar en el hover
            hovertemplate=(
                '<b style="color:black;">' + selected_surface + '</b><br>'  # Nombre de la unidad en negrita y negro
                + 'X: %{x}<br>'
                + 'Y: %{y}<br>'
                + 'Z: %{z}<br>'
                + '<extra></extra>'  # Esto elimina el texto adicional que Plotly agrega por defecto (nombre de la traza)
            ),
            hoverlabel=dict(
                font=dict(
                    color='black'  # Color negro para el texto en el hover
                )
            )
            
            
            
        ))
    # 2. Procesar el archivo TIFF de Oil Top si el botón es presionado
    if oil_top_clicks % 2 != 0:  # Si el número de clics es impar, mostrar la capa
                   
        fig.add_trace(go.Mesh3d(
            x=x_oil_top,
            y=y_oil_top,
            z=z_oil_top,
                
            color='green',  # Color verde fijo
            opacity=opaciti_3d,  # Menos opacidad para la visualización de fondo
            showscale=False,  # No usar escala de colores
            intensity=z_oil_top,  # Usamos la profundidad para determinar la intensidad
            colorscale='Greens',  # Colormap verde
            #name="Oil Top",
            hoverinfo='x+y+z',
            hovertemplate=(
                '<b style="color:black;">' + 'Oil Top' + '</b><br>'
                + 'X: %{x}<br>'
                + 'Y: %{y}<br>'
                + 'Z: %{z}<br>'
                + '<extra></extra>'
            )
        ))

    # 3. Procesar el archivo TIFF de Water Contact si el botón es presionado
    if water_contact_clicks % 2 != 0:  # Si el número de clics es impar, mostrar la capa
        fig.add_trace(go.Mesh3d(
            x=x_water_contact,
            y=y_water_contact,
            z=z_water_contact,
            intensity=z_water_contact,  # Usamos la profundidad para determinar la intensidad
            colorscale='Blues',  # Colormap azul        
            color='blue',  # Color verde fijo
            opacity=opaciti_3d,  # Menos opacidad para la visualización de fondo
            showscale=False,  # No usar escala de colores
                
            #name="Oil Top",
            hoverinfo='x+y+z',
            hovertemplate=(
                '<b style="color:black;">' + 'Water Contact' + '</b><br>'
                + 'X: %{x}<br>'
                + 'Y: %{y}<br>'
                + 'Z: %{z}<br>'
                + '<extra></extra>'
            )
        ))    

    # Solo establecer la posición inicial de la cámara si no se ha hecho aún
    if not camera_initialized:
        camera_initialized = True  # Marcar que la cámara ha sido configurada

           
            
    #configurar escalas    
    if vertical_exaggeration==1:
        rm="data"
    else:
        rm=None
        
    # Definir una lista de colores para los pozos (puedes ampliar esta lista si tienes más pozos)
    well_colors = [
    'red', 'green', 'blue', 'orange', 'purple', 
    'cyan', 'brown', 'pink', 'gray',
    'yellow', 'lime', 'teal', 'violet'
    ]

    # **Pozos**
    if selected_wells:
        for i,well_name in enumerate(selected_wells):
            # Ruta al archivo del pozo
            well_file = os.path.join(wells_path, f"{well_name}")  # Ajusta si los archivos no usan el nombre exacto
            
            # Leer el encabezado para obtener el WELL TYPE
            well_type = "Unknown"  # Valor por defecto
            with open(well_file, 'r') as f:
                for line in f:
                    if "# WELL TYPE:              " in line:
                        # Separar por múltiples espacios o tabulaciones y extraer el tipo de pozo
                        well_type = line.split(":")[-1].strip()
                        break  # No necesitamos seguir leyendo el archivo después de encontrar el tipo


            # Leer los datos del pozo
            column_names = ["MD", "X", "Y", "Z", "TVD", "DX", "DY", "AZIM_TN", "INCL", "DLS", "AZIM_GN"]
            well_data = pd.read_csv(well_file, sep='\s+', skiprows=17, names=column_names)
            
            well_data["Z"]=-well_data["Z"] #poner datos negativos
            
            # Seleccionar un color para el pozo basado en su índice
            well_color = well_colors[i % len(well_colors)]  # Si hay más pozos que colores, recicla los colores
            
            # Configurar el contenido de la tooltip
            customdata = np.array([
                [well_name, well_type, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"]
                for x, y, z in zip(well_data["X"], well_data["Y"], well_data["Z"])
            ])
            
            
            # Añadir la traza del pozo como línea
            fig.add_trace(go.Scatter3d(
                x=well_data["X"],
                y=well_data["Y"],
                z=-well_data["Z"],  # Negativo para profundidades
                mode='lines',
                name=f"{well_name}",
                line=dict(color=well_color, width=4),  # Estilo de la línea
                customdata=customdata,  # Datos adicionales para el hovertemplate
                hovertemplate=(
                    #"Well Name: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "X: %{customdata[2]}<br>"
                    "Y: %{customdata[3]}<br>"
                    "Depth (Z): %{customdata[4]}<br>"
                    #"<extra></extra>"  # Ocultar la información adicional por defecto
                ),
            
                
            ))
    if faults_button_clicks % 2 != 0:  # Si el número de clics es impar, mostrar la capa
        for file_name in os.listdir(faults_path):
            file_path = os.path.join(faults_path, file_name)

            # Ignorar archivos con extensión (como .xml)
            if os.path.isfile(file_path) and not os.path.splitext(file_name)[1]:
                try:
                    # Leer los datos del archivo
                    data = pd.read_csv(file_path, sep='\s+', header=None, names=["x", "y", "z"])
                    # Reducir la calidad: seleccionar una muestra de puntos
                    data = data.sample(frac=resolution_3d, random_state=42)
                
                    # Añadir la superficie como un Mesh3d
                    fig.add_trace(go.Mesh3d(
                        x=data["x"],
                        y=data["y"],
                        z=data["z"]*-1,
                        color='orange',  # Color para las superficies
                        opacity=opaciti_3d-0.2,
                        #name=file_name  # Nombre del archivo como etiqueta
                        hoverinfo='x+y+z',
                        hovertemplate=(
                            '<b>' + 'Fault' + '</b><br>'
                            + 'X: %{x}<br>'
                            + 'Y: %{y}<br>'
                            + 'Z: %{z}<br>'
                            + '<extra></extra>'
                        )
                                ))
                except Exception as e:
                    print(f"Error al procesar {file_name}: {e}")
    #
    # Configurar diseño del gráfico después de agregar todos los trazos
    fig.update_layout(
        #showlegend=False, # Quitar leyendas de pozos
        margin=dict(l=20, r=20, t=20, b=20),
        scene=dict(  # Configuración de los ejes
            #xaxis = dict(nticks=4, range=[431000,440000],),
            #yaxis = dict(nticks=4, range=[6476000,6482000],),
            #zaxis = dict(nticks=4, range=[-3200,100],),
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth (m)",
            aspectmode=rm, # Hacer el gráfico más ancho
            aspectratio=dict(x=1, y=1, z= 0.2 * vertical_exaggeration), #corregir comportamiento escala #añadir norte
            camera=dict(eye=camera_position), # Posición inicial de la cámara (x, y, z)
                        
            
            
        ),
        #title=f"Superficie Geológica (Exageración: {vertical_exaggeration:.1f}x)",
        legend=dict(
            x=1.05,  # Mueve la leyenda fuera del gráfico, a la derecha
            y=1,  # Posición vertical de la leyenda (1 está en la parte superior)
            traceorder='normal',  # Orden de los elementos en la leyenda
            font=dict(size=12),  # Tamaño de la fuente de la leyenda
            borderwidth=1,  # Ancho del borde de la leyenda (opcional)
            
        ),
        
    )
    
   
    
    return fig


# Callback para actualizar la posición de la cámara
@app.callback(
    Output('camera-position', 'data'),
    [Input('3d-surface-plot', 'relayoutData')]
)
def update_camera_position(relayoutData):
    # Verificar si hay datos de la cámara en relayoutData
    if relayoutData and 'scene.camera' in relayoutData:
        camera_data = relayoutData['scene.camera']
        camera_position = camera_data.get('eye', {})
        return camera_position
    return no_update  # Si no hay datos de la cámara, no actualizar

@app.callback(
    [Output('oil-top-button', 'style'),
     Output('water-contact-button', 'style'),
     Output("faults-button", "style")],
    [Input('oil-top-button', 'n_clicks'),
     Input('water-contact-button', 'n_clicks'),
     Input("faults-button", "n_clicks")]
)
def update_button_style(oil_clicks, water_clicks,fault_clicks):
    # Estilo para el botón de Oil Top
    oil_button_style = {
        'color': 'white',
        'width': '8%',
        'height': '40px',
        'backgroundColor': 'green' if oil_clicks and oil_clicks % 2 != 0 else '#333333',  # Cambia a verde si se hizo clic (n_clicks impar)
        'borderRadius': '30px',
        'padding': '10px',
        'border': '2px solid #555555',
        'marginLeft': 'auto',  # Mueve el botón hacia la derecha
        'marginRight': '0',    # Asegura que no haya espacio extra en la parte derecha
        "marginTop": "15%",
        "marginLeft": "7%", 
    }

    # Estilo para el botón de Water Contact
    water_button_style = {
        'color': 'white',
        'width': '8%',
        'height': '40px',
        'backgroundColor': 'blue' if water_clicks and water_clicks % 2 != 0 else '#333333',  # Cambia a azul si se hizo clic (n_clicks impar)
        'borderRadius': '30px',
        'padding': '10px',
        'border': '2px solid #555555',
        'marginLeft': 'auto',  # Mueve el botón hacia la derecha
        'marginRight': '0',    # Asegura que no haya espacio extra en la parte derecha
    }
     # Estilo para el botón de Water Contact
    fault_button_style = {
        'color': 'white',
        'width': '8%',
        'height': '40px',
        'backgroundColor': 'orange' if fault_clicks and fault_clicks % 2 != 0 else '#333333',  # Cambia a azul si se hizo clic (n_clicks impar)
        'borderRadius': '30px',
        'padding': '10px',
        'border': '2px solid #555555',
        'marginLeft': 'auto',  # Mueve el botón hacia la derecha
        'marginRight': '0',    # Asegura que no haya espacio extra en la parte derecha
    }

    return oil_button_style, water_button_style, fault_button_style


"CODIGO PARA VISUALIZACION 2D"

# Callbacks
@app.callback(
    Output('curve-dropdown', 'options'),
    Output('curve-dropdown', 'value'),
    Input('las-file-dropdown', 'value')
)
def update_curve_options(selected_file):
    if not selected_file:
        return [], None

    file_path = os.path.join(logs_folder, selected_file)
   
    # Cargar archivo
    data, columns = process_las_file(file_path)

    # Excluir DEPTH, DEPT y las curvas no necesarias
    filtered_columns = [col for col in columns if col.upper() not in ['DEPTH', 'DEPT']]
    curve_options = [{'label': col, 'value': col} for col in filtered_columns]

    return curve_options, None
    
@app.callback(
    Output('log-graph-container', 'children'),  # Cambié a 'log-graph-container' aquí
    Input('las-file-dropdown', 'value'),
    Input('curve-dropdown', 'value'),
    Input('toggle-lines-button', 'n_clicks')
)

    

def update_graphs(selected_file, selected_curves, n_clicks):
    draw_lines = n_clicks is not None and n_clicks % 2 != 0
    
    # Manejo de casos en los que no hay archivo o curvas seleccionadas
    if not selected_file or not selected_curves:
        return html.Div("")

    file_path = os.path.join(logs_folder, selected_file)
    data, available_columns = process_las_file(file_path)
    
    missing_columns = [col for col in selected_curves if col not in available_columns]
    if missing_columns:
        return html.Div(f"Curvas faltantes: {', '.join(missing_columns)}")
    
    tops_file_path = os.path.join(tops_folder, f"{selected_file}.txt")
    if os.path.exists(tops_file_path):
        top_data = read_top_file(tops_file_path)
    else:
        top_data = pd.DataFrame(columns=["MD", "Unit"])
        
    # Rango de profundidades del log
    log_min_depth = data['DEPT'].min()
    log_max_depth = data['DEPT'].max()
    
    # Filtrar topes dentro del rango del log
    filtered_tops = top_data[(top_data['MD'] >= log_min_depth) & (top_data['MD'] <= log_max_depth)]


    # Lista de colores para asignar a cada gráfico
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'yellow', 'cyan']
    # Generar un gráfico para cada curva seleccionada
    plots = []

   

    for i, curve in enumerate(selected_curves):
        # Procesar datos para la curva actual
        data, _ = process_las_file(file_path, columns_to_load=['DEPT', curve], sample_interval=samples_logs)

        # Limpiar y suavizar los datos
        data_clean = remove_outliers(data, curve)
        data_smoothed = smooth_data(data_clean, curve)

        # Crear la figura
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_smoothed[curve + '_smoothed'],
                y=data_smoothed['DEPT'],
                mode='lines',
                line=dict(color=colors[i % len(colors)]),
                name=curve
            )
        )

        #Si el botón está activado, agregar líneas horizontales para los tops filtrados
        if draw_lines and not filtered_tops.empty:
            for _, row in filtered_tops.iterrows():
                fig.add_hline(
                    y=row['MD'],  # Profundidad del tope
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=row['Unit'],  # Nombre del tope
                    annotation_position="top right"
                )
                






        # Configurar ejes
        if i > 0:
            leg=None
        else:
            leg="MD(m)"
        fig.update_yaxes(autorange='reversed', title=leg)
        fig.update_xaxes(title=curve,tickformat=".2f")
        fig.update_layout(
            height=500,  # Reducido el tamaño del gráfico
            width=200,  # Reducido el tamaño del gráfico
            margin=dict(l=0, r=0, t=0, b=0)
            
                
        )

        # Añadir el gráfico al contenedor
        plots.append(dcc.Graph(figure=fig, id=f"graph-{curve}", config={
        #'displayModeBar': False,
        'displaylogo': False ,# Oculta la barra de herramientas
        #'showTips': False         # Desactiva los tips al pasar el cursor
    }))

    return html.Div(plots, style={'display': 'flex', 'flexDirection': 'row', 'margin-left': '20%'})  # Truncamos la lista a 5 elementos # Los gráficos se muestran en fila
    
# Callback para alternar la visibilidad de la barra lateral
@app.callback(
    [Output("sidebar", "style"), Output("open-button", "style")],
    [Input("toggle-button", "n_clicks"), Input("open-button", "n_clicks")],
    [State("sidebar-state", "data")],
)
def toggle_sidebar(close_clicks, open_clicks, is_open):
    triggered_id = ctx.triggered_id

    if triggered_id == "toggle-button":
        is_open = False
    elif triggered_id == "open-button":
        is_open = True

    if is_open:
        sidebar_style = {
            "position": "fixed",
            "top": "0",
            "left": "0",
            "height": "100%",
            "width": "290px",
            "overflow-x": "hidden",
            "transition": "0.3s",
            "background-color": "rgb(34, 34, 34)",
        }
        open_button_style = {"display": "none"}
    else:
        sidebar_style = {
            "position": "fixed",
            "top": "0",
            "left": "0",
            "height": "100%",
            "width": "0",
            "overflow-x": "hidden",
            "transition": "0.3s",
            "background-color": "rgb(34, 34, 34)",
        }
        open_button_style = {
            "position": "fixed",
            "left": "0",
            "top": "10px",
            "z-index": "1000",
            "backgroundColor": "#333333",
            "color": "white",
            'border': '2px solid #555555',
            "borderRadius": "5px",
            "padding": "10px",
            "cursor": "pointer",
            "top": "90%",        # Ajusta la posición desde la parte superior
            #"left": "21%",
        }
        
    return sidebar_style, open_button_style

# Callback para actualizar el estado almacenado
@app.callback(
    Output("sidebar-state", "data"),
    [Input("toggle-button", "n_clicks"), Input("open-button", "n_clicks")],
    [State("sidebar-state", "data")],
)
def update_state(close_clicks, open_clicks, is_open):
    triggered_id = ctx.triggered_id
    if triggered_id == "toggle-button":
        return False
    elif triggered_id == "open-button":
        return True
    return is_open




# Ejecutar la app
if __name__ == "__main__":
    app.run_server() #debug=True
    