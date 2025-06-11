import tkinter as tk
from tkinter import ttk
import pandas as pd



def mostrar_dataframe_en_tkinter(df, componente_padre):
    """
    Recibe un pandas DataFrame y un componente padre de Tkinter,
    y devuelve un ttk.Frame que contiene un ttk.Treeview
    para mostrar el contenido del DataFrame con barras de desplazamiento.

    Args:
        df (pd.DataFrame): El DataFrame de pandas a mostrar.
        componente_padre (tk.Widget): El widget de Tkinter al que se asociará
                                       este nuevo frame (ej. una ventana raíz,
                                       otro frame, etc.).

    Returns:
        ttk.Frame: Un Frame de Tkinter que contiene el Treeview y las barras de desplazamiento.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("La entrada debe ser un DataFrame de pandas.")
    
    # Crear un frame principal para contener el Treeview y las barras de desplazamiento
    frame_principal = ttk.Frame(componente_padre)

    # Crear un widget Treeview
    vista_arbol = ttk.Treeview(frame_principal, show="headings")

    # Definir columnas
    nombres_columnas = list(df.columns)
    vista_arbol["columns"] = nombres_columnas

    # Establecer encabezados de columna
    for col_nombre in nombres_columnas:
        vista_arbol.heading(col_nombre, text=col_nombre)
        vista_arbol.column(col_nombre, width=100, anchor="w") # Ajusta el ancho si es necesario

    # Insertar datos
    for indice, fila in df.iterrows():
        vista_arbol.insert("", "end", values=list(fila))

    # Añadir barras de desplazamiento
    barra_despl_vertical = ttk.Scrollbar(frame_principal, orient="vertical", command=vista_arbol.yview)
    barra_despl_horizontal = ttk.Scrollbar(frame_principal, orient="horizontal", command=vista_arbol.xview)
    
    vista_arbol.configure(yscrollcommand=barra_despl_vertical.set, xscrollcommand=barra_despl_horizontal.set)

    # Diseño de cuadrícula para Treeview y barras de desplazamiento dentro del frame_principal
    vista_arbol.grid(row=0, column=0, sticky="nsew")
    barra_despl_vertical.grid(row=0, column=1, sticky="ns")
    barra_despl_horizontal.grid(row=1, column=0, sticky="ew")

    # Configurar pesos de cuadrícula para que el Treeview se expanda con el frame
    frame_principal.grid_rowconfigure(0, weight=1)
    frame_principal.grid_columnconfigure(0, weight=1)

    return frame_principal

if __name__ == "__main__":
    # Crear un DataFrame de ejemplo
    datos_ejemplo = {
        'Nombre': ['Ana', 'Luis', 'Carlos', 'Sofía', 'Pedro', 'Laura', 'Miguel', 'Elena', 'Diego', 'Valeria'],
        'Edad': [25, 30, 22, 35, 28, 40, 32, 29, 24, 31],
        'Ciudad': ['Ciudad de México', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana', 'León', 'Juárez', 'Querétaro', 'Mérida', 'Cancún'],
        'Ocupación': ['Ingeniero', 'Artista', 'Estudiante', 'Doctora', 'Diseñador', 'Gerente', 'Analista', 'Programadora', 'Ventas', 'Marketing'],
        'Salario': [70000, 60000, 30000, 90000, 65000, 80000, 72000, 85000, 55000, 78000]
    }
    df_ejemplo = pd.DataFrame(datos_ejemplo)

    # Crear la ventana principal de Tkinter
    ventana_raiz = tk.Tk()
    ventana_raiz.title("Visualizador de DataFrame de Pandas")
    ventana_raiz.geometry("800x600")

    # Mostrar el DataFrame, pasando la ventana_raiz como componente padre
    frame_df = mostrar_dataframe_en_tkinter(df_ejemplo, ventana_raiz)
    frame_df.pack(fill="both", expand=True, padx=10, pady=10)

    # Ejecutar el bucle de eventos de Tkinter
    ventana_raiz.mainloop()