import tkinter as tk
from tkinter import ttk
import pandas as pd
import customtkinter as ctk

def mostrar_dataframe(df, componente_padre):
    """
    Muestra un DataFrame con barras de desplazamiento, usando pack() para posicionamiento.

    Args:
        df (pd.DataFrame): El DataFrame de pandas a mostrar.
        componente_padre (tk.Widget): El widget padre donde se agregará el display.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("La entrada debe ser un DataFrame de pandas.")
    
    # CTkFrame para la tabla y scrollbars
    frame_tabla_completo = ctk.CTkFrame(componente_padre, fg_color="lightblue") 
    
    # Crear el widget Treeview
    vista_arbol = ttk.Treeview(frame_tabla_completo, show="headings")

    # Definir columnas
    nombres_columnas = list(df.columns)
    vista_arbol["columns"] = nombres_columnas

    # Configurar la columna "#0" para que no ocupe espacio visual
    vista_arbol.column("#0", width=0, stretch=False) 
    
    # Establecer encabezados de columna y un ancho inicial
    for col_nombre in nombres_columnas:
        vista_arbol.heading(col_nombre, text=col_nombre)
        vista_arbol.column(col_nombre, width=120, anchor="w", stretch=False) 

    # Insertar datos del DataFrame
    for indice, fila_df in df.iterrows():
        vista_arbol.insert("", "end", values=list(fila_df))

    # Añadir barras de desplazamiento
    barra_despl_vertical = ttk.Scrollbar(frame_tabla_completo, orient="vertical", command=vista_arbol.yview)
    barra_despl_horizontal = ttk.Scrollbar(frame_tabla_completo, orient="horizontal", command=vista_arbol.xview)
    
    vista_arbol.configure(yscrollcommand=barra_despl_vertical.set, xscrollcommand=barra_despl_horizontal.set)

    # --- Disposición INTERNA usando GRID (dentro del frame_tabla_completo) ---
    vista_arbol.grid(row=0, column=0, sticky="nsew")
    barra_despl_vertical.grid(row=0, column=1, sticky="ns")
    barra_despl_horizontal.grid(row=1, column=0, sticky="ew")

    # Configurar pesos de cuadrícula para expansión
    frame_tabla_completo.grid_rowconfigure(0, weight=1)
    frame_tabla_completo.grid_columnconfigure(0, weight=1)

    # --- Posicionamiento en el componente_padre usando PACK ---
    # pack() agrega el widget al final de los widgets ya empaquetados
    frame_tabla_completo.pack(side="top", anchor="nw", padx=0, pady=0)

    # parent_height = componente_padre.winfo_height()
    # desired_height = int(parent_height * 0.4)

    # # Ajustar la altura del widget
    # frame_tabla_completo.pack(fill=tk.BOTH, expand=True)
    # frame_tabla_completo.configure(height=desired_height)    



def regresar_dataframe(df, componente_padre):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("La entrada debe ser un DataFrame de pandas.")
    
    # CTkFrame para la tabla y scrollbars
    frame_tabla_completo = ctk.CTkFrame(componente_padre, fg_color="lightblue") 
    
    # Crear el widget Treeview
    vista_arbol = ttk.Treeview(frame_tabla_completo, show="headings")

    # Definir columnas
    nombres_columnas = list(df.columns)
    vista_arbol["columns"] = nombres_columnas

    # Configurar la columna "#0" para que no ocupe espacio visual
    vista_arbol.column("#0", width=0, stretch=False) 
    
    # Establecer encabezados de columna y un ancho inicial
    for col_nombre in nombres_columnas:
        vista_arbol.heading(col_nombre, text=col_nombre)
        vista_arbol.column(col_nombre, width=120, anchor="w", stretch=False) 

    # Insertar datos del DataFrame
    for indice, fila_df in df.iterrows():
        vista_arbol.insert("", "end", values=list(fila_df))

    # Añadir barras de desplazamiento
    barra_despl_vertical = ttk.Scrollbar(frame_tabla_completo, orient="vertical", command=vista_arbol.yview)
    barra_despl_horizontal = ttk.Scrollbar(frame_tabla_completo, orient="horizontal", command=vista_arbol.xview)
    
    vista_arbol.configure(yscrollcommand=barra_despl_vertical.set, xscrollcommand=barra_despl_horizontal.set)

    # --- Disposición INTERNA usando GRID (dentro del frame_tabla_completo) ---
    vista_arbol.grid(row=0, column=0, sticky="nsew")
    barra_despl_vertical.grid(row=0, column=1, sticky="ns")
    barra_despl_horizontal.grid(row=1, column=0, sticky="ew")

    # Configurar pesos de cuadrícula para expansión
    frame_tabla_completo.grid_rowconfigure(0, weight=1)
    frame_tabla_completo.grid_columnconfigure(0, weight=1)

    # --- Posicionamiento en el componente_padre usando PACK ---
    # pack() agrega el widget al final de los widgets ya empaquetados
    frame_tabla_completo.pack(side="top", anchor="nw", padx=0, pady=0)

    # parent_height = componente_padre.winfo_height()
    # desired_height = int(parent_height * 0.4)

    # # Ajustar la altura del widget
    # frame_tabla_completo.pack(fill=tk.BOTH, expand=True)
    # frame_tabla_completo.configure(height=desired_height)   

# --- Ejemplo de uso con pack() ---
if __name__ == "__main__":
    class SeccionMock:
        def __init__(self, parent):
            self.panel_contenido = ctk.CTkFrame(parent, fg_color="darkgray") 
            self.panel_contenido.pack(fill="both", expand=True, padx=10, pady=10)

    class MiAplicacion:
        def __init__(self, master):
            self.master = master
            master.title("DataFrame Display con Pack")
            master.geometry("800x500")

            seccion = SeccionMock(master)
            
            self.frame_contenido_secc_2 = ctk.CTkFrame(
                seccion.panel_contenido, fg_color="orange" 
            )
            # Usar pack() en lugar de grid()
            self.frame_contenido_secc_2.pack(side="top", anchor="nw", padx=20, pady=20)

            self.df_pedidos = pd.DataFrame({
                'dia_pe': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'dia_er': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                'especie': ['Malus domestica', 'Musa paradisiaca', 'Daucus carota', 'Zea mays', 'Mangifera indica', 'Allium cepa', 'Solanum tuberosum', 'Solanum lycopersicum', 'Helianthus annuus', 'Ficus carica'],
                'proveedor': ['Laguna Seca', 'Moctezuma', 'Proveedor 4', 'Moctezuma', 'Proveedor 4', 'Proveedor 4', 'Proveedor 4', 'Viveros del Sol', 'Viveros del Sol', 'GrowFast'],
                'cantidad': [3877.0, 2350.0, 3877.0, 3877.0, 4584.0, 3525.0, 3291.0, 5994.0, 3525.0, 4100.0],
                'otra_columna_larga': ['Texto largo para probar expansión', 'Otro texto largo', 'Corto', 'Un poco más largo', 'Texto de prueba', 'Otro texto', 'Corto', 'Más largo', 'Texto extenso', 'Final del texto']
            })

            # La tabla se "append" (agrega) al frame usando pack()
            mostrar_dataframe(self.df_pedidos, self.frame_contenido_secc_2)

    ventana_raiz = ctk.CTk()
    app = MiAplicacion(ventana_raiz)
    ventana_raiz.mainloop()