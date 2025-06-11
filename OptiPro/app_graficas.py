import simulation_function as sim
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def secc_01_muestra_grafica(app):

    if app.canvas_01_demanda:
        app.canvas_01_demanda.get_tk_widget().destroy()

    poligono = app.poligono_actual
    if poligono == 0:
        poligono = None

    app.fig_demanda = sim.plot_demanda_cubierta_por_especie_poligono(
        app.df_cobertura_demanda, app.secc1_especie_actual, poligono
    )
    app.canvas_01_demanda = FigureCanvasTkAgg(
        app.fig_demanda, app.frame_contenido_secc_1
    )
    app.canvas_01_demanda.draw()
    app.canvas_01_demanda.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def secc_01_on_especie_select(app, selected_especie):
    app.secc1_especie_actual = selected_especie
    secc_01_muestra_grafica(app)
    print(f"Especie seleccionada: {selected_especie}")


def secc_01_on_poligono_select(app, selected_poligono):
    app.poligono_actual = selected_poligono
    secc_01_muestra_grafica(app)
    print(f"Polígono seleccionado: {selected_poligono}")


# def secc_02_muestra_grafica(app):
#     if app.canvas_02_pedidos:
#         app.canvas_02_pedidos.get_tk_widget().destroy()

#     app.fig_pedidos = sim.plot_pedidos_por_dia(
#         app.df_pedidos, por=app.secc2_vista_actual
#     )
#     app.canvas_02_pedidos = FigureCanvasTkAgg(
#         app.fig_pedidos, app.frame_contenido_secc_2
#     )
#     app.canvas_02_pedidos.draw()
#     app.canvas_02_pedidos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def secc_02_muestra_grafica(app):
    if app.canvas_02_pedidos:
        app.canvas_02_pedidos.get_tk_widget().destroy()

    app.fig_pedidos = sim.plot_pedidos_por_dia(
        app.df_pedidos, por=app.secc2_vista_actual
    )
    app.canvas_02_pedidos = FigureCanvasTkAgg(
        app.fig_pedidos, app.frame_contenido_secc_2
    )
    app.canvas_02_pedidos.draw()

    # Obtener la altura del parent y calcular el 60%
    parent_height = app.frame_contenido_secc_2.winfo_height()
    desired_height = int(parent_height * 0.6)

    # Ajustar la altura del widget
    canvas_widget = app.canvas_02_pedidos.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas_widget.config(height=desired_height)



def secc_02_on_vista_select(app, selected_vista):
    app.secc2_vista_actual = selected_vista
    secc_02_muestra_grafica(app)
    print(f"Vista seleccionada: {selected_vista}")


def secc_03_muestra_grafica(app):
    if app.canvas_03_inventario:
        app.canvas_03_inventario.get_tk_widget().destroy()

    app.fig_inventario = sim.plot_inventario_total_diario_por_especie(
        app.df_inventario_diario, app.secc3_especie_actual
    )
    app.canvas_03_inventario = FigureCanvasTkAgg(
        app.fig_inventario, app.frame_contenido_secc_3
    )
    app.canvas_03_inventario.draw()
    app.canvas_03_inventario.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def secc_03_on_especie_select(app, selected_especie):
    app.secc3_especie_actual = selected_especie
    secc_03_muestra_grafica(app)
    print(f"Especie seleccionada: {selected_especie}")


def secc_04_muestra_grafica(app):
    if app.canvas_04_rutas:
        app.canvas_04_rutas.get_tk_widget().destroy()

    dia = app.secc4_dia_var.get()
    app.fig_rutas = sim.plot_rutas_networkx_por_dia(app.df_rutas, dia)
    app.canvas_04_rutas = FigureCanvasTkAgg(app.fig_rutas, app.frame_contenido_secc_4)
    app.canvas_04_rutas.draw()
    app.canvas_04_rutas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def secc_04_on_dia_select(app, app_instance):
    secc_04_muestra_grafica(app)


def secc_05_muestra_grafica(app):
    if app.canvas_05_costos:
        app.canvas_05_costos.get_tk_widget().destroy()

    app.fig_inventario = sim.plot_costos_totales_por_dia(app.df_costos_solucion)
    app.canvas_05_costos = FigureCanvasTkAgg(
        app.fig_inventario, app.frame_contenido_secc_5
    )
    app.canvas_05_costos.draw()
    app.canvas_05_costos.get_tk_widget().pack(fill=tk.BOTH, expand=True)
