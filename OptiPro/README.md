# OptiPro - Sistema de Optimización Logística

OptiPro es una aplicación de escritorio diseñada para optimizar la logística de distribución de plantas desde viveros hasta sitios de plantación. La aplicación proporciona una interfaz gráfica moderna y funcional para gestionar todo el proceso logístico.

## Características Principales

### 1. Gestión de Inventario
- Control y seguimiento de plantas disponibles en vivero
- Monitoreo de tiempos de aclimatación
- Optimización del espacio en vivero
- Control de capacidad máxima de almacenamiento

### 2. Planificación de Rutas
- Generación automática de rutas óptimas de distribución
- Consideración de restricciones de tiempo y capacidad
- Balanceo eficiente de cargas por ruta
- Optimización de tiempos de entrega

### 3. Análisis de Demanda
- Visualización detallada de demanda por especie
- Seguimiento en tiempo real de cobertura de demanda
- Planificación inteligente de pedidos
- Gestión de proveedores

### 4. Optimización de Recursos
- Minimización de costos de transporte
- Maximización de eficiencia en entregas
- Gestión óptima de tiempos de aclimatación
- Control de capacidad de carga y tiempos máximos

## Requisitos del Sistema

La aplicación requiere:
- Python 3.x
- Bibliotecas especificadas en `requirements.txt`

## Instalación

1. Clone el repositorio
2. Cree un entorno virtual:
   ```bash
   python -m venv .venv
   ```
3. Active el entorno virtual:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Instale las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Prepare su archivo de parámetros en Excel con las siguientes hojas:
   - parametros: Configuraciones generales
   - hectareas_poligono: Área por polígono
   - demanda_especies: Demanda por especie
   - costos_proveedores: Matriz de costos
   - distancias_poligonos: Matriz de distancias

2. Ejecute la aplicación:
   ```bash
   python app.py
   ```

3. Use la interfaz gráfica para:
   - Importar datos desde Excel
   - Visualizar demanda y cobertura
   - Monitorear inventario
   - Optimizar rutas de distribución

## Estructura del Proyecto

- `app.py`: Aplicación principal y GUI
- `app_ui.py`: Componentes de la interfaz de usuario
- `simulation_function.py`: Lógica de simulación y optimización
- `requirements.txt`: Dependencias del proyecto
- `simulation_notebook.py`: Copia del archivo original de notebook Lógica de simulación y optimización


## Cambios de simulation_notebook.py (Reto.ipynb) hacia simulation_function.py

1. Se separó la carga de parámetros en una nueva función `preparacion(archivo)` y se creó `run_simulation_external(archivo)` como punto de entrada principal
2. Los archivos de parámetros ahora se buscan en la carpeta "escenarios/"
2. Las funciones de visualización ahora retornan las figuras en lugar de mostrarlas, removiendo los `plt.show()` automáticos
3. Se modificó el retorno de la simulación para incluir un diccionario completo con figuras (rutas, demanda, inventario), datos de simulación y históricos de órdenes/distribuciones
5. Se agregó sistema de logs con timestamps usando `cprint` y una lista global `console_messages` para guardar el historial de mensajes y mostrarlo luego en la UI


## Crear Ejecutable

Borrar dist\ build\

pip install pyinstaller
pyinstaller optipro.spec

(No ejecutar: pyinstaller --onefile --name optipro --add-data "img;img" --icon="img/optipro.ico" --windowed app.py)

## Desarrollado por

Leonardo De Regil Cárdenas, Eduardo Mitrani Krouham, Victor Adid Salgado Santana, Mario Alberto Landa Flores
