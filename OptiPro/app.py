# app.py - Aplicación principal que gestiona la interfaz gráfica y la interacción con el usuario para el sistema de optimización logística OptiPro
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image
import os, sys
from app_ui import DialogoAcerca, SeccionContenido, ColoresUI
import simulation_function as sim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import shutil


class Aplicacion(ctk.CTk):
    """Ventana principal de la aplicación que gestiona la interfaz y navegación"""

    def __init__(self):
        super().__init__()

        # Configurar protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables de estado de simulación
        self.simulation_data = None
        self.excel_file = None
        self.simulation_running = False

        # Configurar ventana
        self.title("OptiPro")
        self.geometry("1000x600")
        self.minsize(800, 500)  # Tamaño mínimo para evitar problemas de layout

        # Configurar grid principal de la ventana
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Barra lateral
        self.grid_columnconfigure(1, weight=0)  # Barra de colapso
        self.grid_columnconfigure(2, weight=1)  # Contenido principal

        # Inicializar colores
        self.colores = ColoresUI()
        self.color_activo = self.colores.activo
        self.color_inactivo = self.colores.inactivo
        self.color_hover = self.colores.hover
        self.color_barra_colapso = self.colores.barra_colapso
        self.color_barra_hover = self.colores.barra_hover
        self.color_fondo_barra_lateral = self.colores.fondo_barra_lateral
        self.color_fondo_contenido = self.colores.fondo_contenido
        self.color_texto_menu = self.colores.texto_menu
        self.color_fondo_paneles = self.colores.fondo_paneles
        self.color_titulo = self.colores.titulo
        self.color_acerca = self.colores.acerca

        # Estado de la barra lateral
        self.barra_lateral_expandida = True
        self.ancho_barra_lateral_expandida = 160
        self.ancho_barra_lateral_colapsada = 64
        self.ancho_barra_colapso = 30
        self.ancho_botones = 130  # Ancho fijo para botones y logo

        # Configurar color de fondo de la ventana principal
        self.configure(fg_color=self.color_fondo_contenido)

        # Crear marco principal para el contenido
        self.marco_principal = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.marco_principal.grid(row=0, column=2, sticky="nsew")
        self.marco_principal.grid_rowconfigure(
            0, weight=1
        )  # Todo el espacio disponible
        self.marco_principal.grid_columnconfigure(0, weight=1)

        # Configurar grid de la ventana principal para el panel inferior
        self.grid_rowconfigure(0, weight=1)  # Contenido principal
        self.grid_rowconfigure(1, weight=0)  # Panel inferior (altura fija)

        # Crear área de consola (panel inferior)
        self.frame_consola = ctk.CTkFrame(
            self,  # Ahora es hijo de la ventana principal
            fg_color=self.color_fondo_paneles,
            corner_radius=10,
            height=100,  # Altura fija
        )
        self.frame_consola.grid(row=1, column=2, sticky="ew", padx=10, pady=(0, 10))
        self.frame_consola.grid_propagate(False)  # Mantener altura fija
        self.frame_consola.grid_rowconfigure(0, weight=1)
        self.frame_consola.grid_columnconfigure(0, weight=1)

        # Widget de texto para la consola
        self.consola = ctk.CTkTextbox(
            self.frame_consola,
            wrap="word",
            font=("Consolas", 12),
            fg_color=self.color_fondo_paneles,
            text_color="white",
            height=80,  # Ligeramente menor que el frame para los paddings
        )
        self.consola.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Ocultar la consola inicialmente
        self.frame_consola.grid_remove()

        # Configurar protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Cargar íconos
        self.cargar_imagenes()

        # Crear marco de barra lateral
        self.marco_barra_lateral = ctk.CTkFrame(
            self,
            width=self.ancho_barra_lateral_expandida,
            corner_radius=0,
            fg_color=self.color_fondo_barra_lateral,
        )
        self.marco_barra_lateral.grid(row=0, column=0, sticky="nsew")
        self.marco_barra_lateral.grid_propagate(False)

        # Configurar filas del sidebar
        self.marco_barra_lateral.grid_rowconfigure(0, weight=0)  # Logo
        self.marco_barra_lateral.grid_rowconfigure(1, weight=0)  # Nombre app
        self.marco_barra_lateral.grid_rowconfigure(2, weight=1)  # Espacio botones
        self.marco_barra_lateral.grid_rowconfigure(3, weight=0)  # Botón acerca

        # Frame contenedor del logo para centrado
        self.frame_logo = ctk.CTkFrame(
            self.marco_barra_lateral,
            height=80,  # Ajustado a la altura exacta del logo
            width=160,  # Ancho fijo inicial
            fg_color="transparent",
            corner_radius=0,
        )
        self.frame_logo.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.frame_logo.grid_propagate(False)

        # Logo clickeable
        self.label_logo = ctk.CTkButton(
            self.frame_logo,
            text="",  # Sin texto, usaremos imagen
            image=self.logo_expandido,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            corner_radius=0,
            height=80,
            width=160,
            command=self.mostrar_inicio,
        )
        self.label_logo.place(relx=0.5, rely=0.5, anchor="center")  # Centrado perfecto

        # Crear frame para los botones
        self.frame_botones = ctk.CTkFrame(
            self.marco_barra_lateral, fg_color="transparent", corner_radius=0
        )
        self.frame_botones.grid(row=1, column=0, sticky="nsew", padx=0)
        self.frame_botones.grid_columnconfigure(0, weight=1)

        # Botones de la barra lateral con íconos
        self.boton_seccion1 = ctk.CTkButton(
            self.frame_botones,
            text="Demanda",
            image=self.imagen_seccion1,
            compound="left",
            anchor="w",
            height=40,
            corner_radius=0,
            command=self.mostrar_seccion1,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            text_color=self.color_texto_menu,
            state="disabled",  # Inicialmente deshabilitado
        )
        self.boton_seccion1.grid(row=0, column=0, pady=5, sticky="ew", padx=0)

        self.boton_seccion2 = ctk.CTkButton(
            self.frame_botones,
            text="Inventario",
            image=self.imagen_seccion2,
            compound="left",
            anchor="w",
            height=40,
            corner_radius=0,
            command=self.mostrar_seccion2,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            text_color=self.color_texto_menu,
            state="disabled",  # Inicialmente deshabilitado
        )
        self.boton_seccion2.grid(row=1, column=0, pady=5, sticky="ew", padx=0)

        self.boton_seccion3 = ctk.CTkButton(
            self.frame_botones,
            text="Rutas",
            image=self.imagen_seccion3,
            compound="left",
            anchor="w",
            height=40,
            corner_radius=0,
            command=self.mostrar_seccion3,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            text_color=self.color_texto_menu,
            state="disabled",  # Inicialmente deshabilitado
        )
        self.boton_seccion3.grid(row=2, column=0, pady=5, sticky="ew", padx=0)

        # Botón Acerca de la Aplicación
        self.boton_seccion9 = ctk.CTkButton(
            self.marco_barra_lateral,
            text="Acerca de",
            image=self.imagen_seccion9,
            compound="left",
            font=ctk.CTkFont(size=13),
            height=32,
            fg_color=self.color_inactivo,
            text_color=self.color_texto_menu,
            hover_color=self.color_hover,
            corner_radius=0,
            anchor="w",
            command=self.mostrar_acerca,
        )
        self.boton_seccion9.grid(row=3, column=0, pady=0, sticky="ew", padx=0)

        # Barra de colapso (usando Label en lugar de Button)
        self.barra_colapso = ctk.CTkLabel(
            self,
            text="",
            width=15,
            height=self.winfo_height(),
            corner_radius=0,
            fg_color=self.color_barra_colapso,
            font=ctk.CTkFont(size=14),
            anchor="center",
        )
        self.barra_colapso.grid(row=0, column=1, sticky="ns")

        # Configurar la columna para que mantenga el ancho mínimo
        self.grid_columnconfigure(1, weight=0, minsize=15)

        # Bind eventos de mouse para la barra
        self.barra_colapso.bind("<Enter>", self.al_entrar_barra)
        self.barra_colapso.bind("<Leave>", self.al_salir_barra)
        self.barra_colapso.bind("<Button-1>", self.alternar_barra_lateral)

        # Hacer que el cursor cambie al pasar por encima
        self.barra_colapso.configure(cursor="hand2")

        # Almacenar botones en un diccionario para fácil acceso
        self.botones_menu = {
            "seccion1": self.boton_seccion1,
            "seccion2": self.boton_seccion2,
            "seccion3": self.boton_seccion3,
        }

        # Inicializar secciones de contenido
        self.secciones = {}
        self.crear_secciones()

        # Mostrar la sección de inicio por defecto
        self.mostrar_inicio()

    # === ÁREA: CARGA DE RECURSOS E IMÁGENES ===
    def cargar_imagenes(self):
        """Carga todos los recursos de imágenes necesarios para la interfaz"""
        
        archivo = resource_path("img/logo1.png")
        self.logo_expandido = ctk.CTkImage(            
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(160, 80),  # Tamaño para modo expandido
        )
        archivo = resource_path("img/logo2.png")
        self.logo_colapsado = ctk.CTkImage(            
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(64, 80),  # Tamaño cuadrado para modo colapsado
        )
        archivo = resource_path("img/seccion1.png")
        self.imagen_seccion1 = ctk.CTkImage(
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(24, 24),
        )
        archivo = resource_path("img/seccion2.png")
        self.imagen_seccion2 = ctk.CTkImage(
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(24, 24),
        )
        archivo = resource_path("img/seccion3.png")
        self.imagen_seccion3 = ctk.CTkImage(
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(24, 24),
        )
        archivo = resource_path("img/seccion9.png")
        self.imagen_seccion9 = ctk.CTkImage(
            light_image=Image.open(archivo),
            dark_image=Image.open(archivo),
            size=(24, 24),
        )

    # === ÁREA: MANEJO DE LA BARRA LATERAL Y COLAPSO ===
    def al_entrar_barra(self, event):
        """Maneja el efecto hover al entrar el mouse en la barra de colapso"""
        self.barra_colapso.configure(fg_color=self.color_barra_hover)

    def al_salir_barra(self, event):
        """Maneja el efecto hover al salir el mouse de la barra de colapso"""
        self.barra_colapso.configure(fg_color=self.color_barra_colapso)

    def alternar_barra_lateral(self, event=None):
        """Alterna el estado expandido/colapsado de la barra lateral"""
        if self.barra_lateral_expandida:
            # Colapsar
            nueva_anchura = self.ancho_barra_lateral_colapsada
            self.marco_barra_lateral.configure(width=nueva_anchura)

            # Ajustar botones en modo colapsado
            self.boton_seccion1.configure(text="", image=self.imagen_seccion1)
            self.boton_seccion2.configure(text="", image=self.imagen_seccion2)
            self.boton_seccion3.configure(text="", image=self.imagen_seccion3)
            self.boton_seccion9.configure(text="", image=self.imagen_seccion9)

            # Ajustar logo en modo colapsado
            self.label_logo.configure(image=self.logo_colapsado, width=64, height=80)
            self.frame_logo.configure(width=64)
            self.label_logo.grid_configure(padx=0)
            self.barra_lateral_expandida = False
        else:
            # Expandir
            nueva_anchura = self.ancho_barra_lateral_expandida
            self.marco_barra_lateral.configure(width=nueva_anchura)

            # Restaurar botones expandidos
            self.boton_seccion1.configure(text="Demanda", image=self.imagen_seccion1)
            self.boton_seccion2.configure(text="Inventario", image=self.imagen_seccion2)
            self.boton_seccion3.configure(text="Rutas", image=self.imagen_seccion3)
            self.boton_seccion9.configure(text="Info", image=self.imagen_seccion9)

            # Restaurar logo expandido
            self.label_logo.configure(image=self.logo_expandido, width=160, height=80)
            self.frame_logo.configure(width=160)
            self.label_logo.grid_configure(padx=0)
            self.barra_lateral_expandida = True

        # Actualizar el estado activo de los botones
        self.actualizar_botones_menu(
            self.opcion_activa if hasattr(self, "opcion_activa") else None
        )

    # === ÁREA: GESTIÓN DE NAVEGACIÓN Y SECCIONES ===
    def crear_secciones(self):
        """Inicializa todas las secciones de contenido de la aplicación"""
        # Crear todas las secciones de contenido
        self.secciones["inicio"] = SeccionContenido(
            self.marco_principal,
            "OptiPro",
            self.color_fondo_contenido,
            self.color_titulo,
        )
        self.secciones["seccion1"] = SeccionContenido(
            self.marco_principal,
            "Demanda",
            self.color_fondo_contenido,
            self.color_titulo,
        )
        self.secciones["seccion2"] = SeccionContenido(
            self.marco_principal,
            "Inventario",
            self.color_fondo_contenido,
            self.color_titulo,
        )
        self.secciones["seccion3"] = SeccionContenido(
            self.marco_principal,
            "Rutas",
            self.color_fondo_contenido,
            self.color_titulo,
        )

        # Configurar cada sección para que ocupe todo el espacio
        for seccion in self.secciones.values():
            seccion.grid_columnconfigure(0, weight=1)
            seccion.grid_columnconfigure(1, weight=1)

        # Configurar la sección de inicio
        seccion = self.secciones["inicio"]
        seccion.etiqueta_subtitulo.configure(
            text="Sistema de Optimización Logística para Distribución de Plantas"
        )

        # Configurar el grid del panel de contenido para dos columnas (1/3 y 2/3)
        seccion.panel_contenido.grid_columnconfigure(
            0, weight=1
        )  # Panel izquierdo (1/3)
        seccion.panel_contenido.grid_columnconfigure(1, weight=2)  # Panel derecho (2/3)

        # Crear un frame contenedor para mantener la proporción 1/3
        frame_izquierdo = ctk.CTkFrame(seccion.panel_contenido, fg_color="transparent")
        frame_izquierdo.grid(row=0, column=0, sticky="nsew")
        frame_izquierdo.grid_columnconfigure(0, weight=1)
        frame_izquierdo.grid_rowconfigure(0, weight=1)

        # Panel izquierdo - Bienvenida (centrado en el frame_izquierdo)
        frame_bienvenida = ctk.CTkFrame(frame_izquierdo, fg_color="transparent")
        frame_bienvenida.place(relx=0.5, rely=0.5, anchor="center")

        # Mensaje de bienvenida
        ctk.CTkLabel(
            frame_bienvenida,
            text="OptiPro",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color=self.color_titulo,
        ).pack(pady=10)

        ctk.CTkLabel(
            frame_bienvenida,
            text="Optimización Logística\npara la Distribución de Plantas",
            font=ctk.CTkFont(size=24),
            text_color="gray60",
        ).pack(pady=10)

        # Frame para los botones
        frame_botones = ctk.CTkFrame(frame_bienvenida, fg_color="transparent")
        frame_botones.pack(pady=30)

        # Botón para importar archivo
        self.btn_importar = ctk.CTkButton(
            frame_botones,
            text="Importar Excel",
            command=self.importar_excel,
            width=200,
            height=40,
            fg_color=self.color_activo,
            hover_color=self.color_hover,
            text_color="white",
            font=ctk.CTkFont(size=16),
            state="normal",  # Inicialmente activo
        )
        self.btn_importar.pack(pady=(0, 10))

        # Botón para iniciar optimización
        self.btn_optimizar = ctk.CTkButton(
            frame_botones,
            text="Iniciar Optimización",
            command=self.iniciar_optimizacion,
            width=200,
            height=40,
            fg_color=self.color_activo,
            hover_color=self.color_hover,
            text_color="white",
            font=ctk.CTkFont(size=16),
            state="disabled",  # Inicialmente inactivo
        )
        self.btn_optimizar.pack(pady=(0, 10))

        # Botón para reiniciar
        self.btn_reiniciar = ctk.CTkButton(
            frame_botones,
            text="Reiniciar",
            command=self.reiniciar_sistema,
            width=200,
            height=40,
            fg_color=self.color_activo,
            hover_color=self.color_hover,
            text_color="white",
            font=ctk.CTkFont(size=16),
            state="disabled",  # Inicialmente inactivo
        )
        self.btn_reiniciar.pack()

        # Panel derecho - Consola
        self.frame_consola = ctk.CTkFrame(
            seccion.panel_contenido, fg_color="black", corner_radius=0
        )
        self.frame_consola.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.frame_consola.grid_rowconfigure(0, weight=1)
        self.frame_consola.grid_columnconfigure(0, weight=1)
        self.frame_consola.grid_remove()  # Ocultar inicialmente

        # Widget de texto para la consola
        self.consola = ctk.CTkTextbox(
            self.frame_consola,
            wrap="word",
            font=("Consolas", 12),
            fg_color="#F3F3F3",
            text_color="#006AB1",
            corner_radius=0,
        )
        self.consola.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Mostrar nombre del archivo Excel en todas las secciones
        for seccion in self.secciones.values():
            if self.excel_file:
                seccion.actualizar_estado(f"Archivo cargado: {self.excel_file}")
            else:
                seccion.actualizar_estado("Selecciona un archivo Excel")

        # Configurar sección 1 (Demanda)
        seccion = self.secciones["seccion1"]

        # Actualizar subtítulo
        seccion.etiqueta_subtitulo.configure(
            text="Análisis y seguimiento de demanda por especie"
        )

        # Crear frame para la gráfica
        frame_grafica = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
        frame_grafica.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Mensaje inicial
        ctk.CTkLabel(
            frame_grafica,
            text="Ejecute la optimización para ver los resultados",
            font=ctk.CTkFont(size=16),
            text_color="gray60",
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Configurar sección 2 (Inventario)
        seccion = self.secciones["seccion2"]

        # Actualizar subtítulo
        seccion.etiqueta_subtitulo.configure(
            text="Control y monitoreo del inventario en vivero"
        )

        # Crear frame para la gráfica
        frame_grafica = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
        frame_grafica.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Mensaje inicial
        ctk.CTkLabel(
            frame_grafica,
            text="Ejecute la optimización para ver los resultados",
            font=ctk.CTkFont(size=16),
            text_color="gray60",
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Configurar sección 3 (Rutas)
        seccion = self.secciones["seccion3"]

        # Actualizar subtítulo
        seccion.etiqueta_subtitulo.configure(
            text="Planificación y optimización de rutas de distribución"
        )

        # Crear frame para la gráfica
        self.frame_rutas = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
        self.frame_rutas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Mensaje inicial
        ctk.CTkLabel(
            self.frame_rutas,
            text="Ejecute la optimización para ver los resultados",
            font=ctk.CTkFont(size=16),
            text_color="gray60",
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Configurar el grid del panel de contenido
        seccion.panel_contenido.grid_rowconfigure(0, weight=1)
        seccion.panel_contenido.grid_columnconfigure(0, weight=1)

        # Crear frame para controles de navegación en el panel superior
        frame_controles = ctk.CTkFrame(
            seccion.marco_notificaciones, fg_color="transparent"
        )
        frame_controles.grid(row=0, column=0, sticky="e")

        # Botón anterior
        self.btn_anterior = ctk.CTkButton(
            frame_controles,
            text="← Anterior",
            command=self.dia_anterior,
            width=80,
            height=24,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            font=ctk.CTkFont(size=12),
            state="disabled",
        )
        self.btn_anterior.grid(row=0, column=0, padx=5)

        # Etiqueta del día actual
        self.lbl_dia = ctk.CTkLabel(
            frame_controles,
            text="Día -",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.color_titulo,
        )
        self.lbl_dia.grid(row=0, column=1, padx=10)

        # Botón siguiente
        self.btn_siguiente = ctk.CTkButton(
            frame_controles,
            text="Siguiente →",
            command=self.dia_siguiente,
            width=80,
            height=24,
            fg_color=self.color_inactivo,
            hover_color=self.color_hover,
            font=ctk.CTkFont(size=12),
            state="disabled",
        )
        self.btn_siguiente.grid(row=0, column=2, padx=5)

        # Inicializar variables de control
        self.dia_actual = 0
        self.total_dias = 0

    def dia_anterior(self):
        """Navega al día anterior"""
        if self.dia_actual > 0:
            self.dia_actual -= 1
            self.mostrar_ruta_actual()

    def dia_siguiente(self):
        """Navega al día siguiente"""
        if self.dia_actual < self.total_dias - 1:
            self.dia_actual += 1
            self.mostrar_ruta_actual()

    def mostrar_ruta_actual(self):
        """Muestra la ruta del día actual"""
        # Limpiar el frame de rutas
        for widget in self.frame_rutas.winfo_children():
            widget.destroy()

        # Actualizar etiqueta del día
        self.lbl_dia.configure(text=f"Día {self.dia_actual + 1}")

        # Actualizar estado de los botones
        self.btn_anterior.configure(
            state="normal" if self.dia_actual > 0 else "disabled"
        )
        self.btn_siguiente.configure(
            state="normal" if self.dia_actual < self.total_dias - 1 else "disabled"
        )

        # Mostrar la figura actual
        if self.dia_actual < len(self.fig_rutas):
            fig = self.fig_rutas[self.dia_actual]
            canvas = FigureCanvasTkAgg(fig, self.frame_rutas)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def mostrar_seccion(self, nombre_seccion):
        """Muestra la sección especificada y actualiza el estado de los botones"""
        # Ocultar todas las secciones
        for seccion in self.secciones.values():
            seccion.grid_remove()

        # Mostrar sección seleccionada
        self.secciones[nombre_seccion].grid(row=0, column=0, sticky="nsew")

        # Actualizar el estado de los botones del menú
        self.opcion_activa = nombre_seccion
        self.actualizar_botones_menu(nombre_seccion)

    def actualizar_botones_menu(self, opcion_activa):
        """Actualiza el estado visual de los botones según la sección activa"""
        # Actualizar el estado visual de todos los botones
        for opcion, boton in self.botones_menu.items():
            if opcion == opcion_activa:
                boton.configure(fg_color=self.color_activo)
            else:
                boton.configure(fg_color=self.color_inactivo)

    def mostrar_seccion1(self):
        """Muestra el contenido de la sección 1"""
        self.mostrar_seccion("seccion1")

    def mostrar_seccion2(self):
        """Muestra el contenido de la sección 2"""
        self.mostrar_seccion("seccion2")

    def mostrar_seccion3(self):
        """Muestra el contenido de la sección 3"""
        self.mostrar_seccion("seccion3")

    def mostrar_inicio(self):
        """Muestra la sección de inicio"""
        self.mostrar_seccion("inicio")

    # === ÁREA: DIÁLOGOS Y VENTANAS ADICIONALES ===
    def mostrar_acerca(self):
        """Muestra el diálogo con información sobre la aplicación"""
        dialogo = DialogoAcerca(self)
        self.wait_window(dialogo)

    def iniciar_optimizacion(self):
        """Inicia el proceso de optimización"""
        # Mostrar la consola inmediatamente
        self.frame_consola.grid()
        self.frame_consola.update()  # Forzar actualización inmediata

        if not self.simulation_running:
            self.simulation_running = True

            # Cambiar el botón a estado de procesamiento
            self.btn_optimizar.configure(
                text="Procesando...",
                fg_color="#FFA500",  # Naranja
                text_color="#000000",  # Negro puro
                state="disabled",
            )
            self.btn_optimizar.update()  # Forzar actualización del botón

            # Limpiar la consola
            self.consola.delete("1.0", tk.END)

            try:
                # Ejecutar la simulación pasando el archivo Excel
                self.simulation_data = sim.run_simulation_external(self.excel_file)

                # Mostrar los mensajes acumulados
                for line in sim.console_messages:
                    self.consola.insert(tk.END, line)
                self.consola.see(tk.END)  # Scroll al final

                # Habilitar botones
                self.boton_seccion1.configure(state="normal")
                self.boton_seccion2.configure(state="normal")
                self.boton_seccion3.configure(state="normal")

                # Actualizar gráficas pero sin cambiar de sección
                self.actualizar_graficas()

                # Actualizar estado de los botones después de optimizar
                self.btn_optimizar.configure(
                    text="Completada",
                    fg_color=self.color_activo,
                    text_color="white",
                    state="disabled",
                )
                self.btn_importar.configure(state="disabled")
                self.btn_reiniciar.configure(state="normal")

            except Exception as e:
                self.consola.insert(
                    tk.END, f"\nError durante la simulación: {str(e)}\n"
                )
                self.consola.see(tk.END)

                # Cambiar el botón a estado de error
                self.btn_optimizar.configure(
                    text="Error - Intentar de nuevo",
                    fg_color="#DC3545",  # Rojo
                    text_color="white",  # Blanco
                    state="normal",
                )
            finally:
                self.simulation_running = False

    def actualizar_graficas(self):
        """Actualiza las gráficas con los resultados de la simulación"""
        if self.simulation_data:
            # Actualizar sección 1 (Demanda)
            seccion = self.secciones["seccion1"]
            for widget in seccion.panel_contenido.winfo_children():
                widget.destroy()

            frame_grafica = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
            frame_grafica.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

            fig_demanda = self.simulation_data["fig_demanda"]
            canvas = FigureCanvasTkAgg(fig_demanda, frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Actualizar sección 2 (Inventario)
            seccion = self.secciones["seccion2"]
            for widget in seccion.panel_contenido.winfo_children():
                widget.destroy()

            frame_grafica = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
            frame_grafica.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

            fig_inventario = self.simulation_data["fig_inventario"]
            canvas = FigureCanvasTkAgg(fig_inventario, frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Actualizar sección 3 (Rutas)
            seccion = self.secciones["seccion3"]
            for widget in seccion.panel_contenido.winfo_children():
                widget.destroy()

            self.frame_rutas = ctk.CTkFrame(seccion.panel_contenido, fg_color="white")
            self.frame_rutas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

            self.fig_rutas = self.simulation_data["fig_rutas"]
            self.total_dias = len(self.fig_rutas)
            self.dia_actual = 0
            self.mostrar_ruta_actual()

    def on_closing(self):
        """Maneja el cierre limpio de la aplicación"""
        plt.close("all")  # Cerrar todas las figuras de matplotlib
        self.quit()  # Cerrar la ventana principal

    def importar_excel(self):
        """Permite al usuario seleccionar un archivo Excel para importar"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo Excel",
            filetypes=[("Archivos Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if filename:
            try:
                # Crear carpeta de entrada si no existe
                if not os.path.exists("escenarios"):
                    os.makedirs("escenarios")

                # Copiar archivo a la carpeta de entrada
                nombre_archivo = os.path.basename(filename)
                destino = os.path.join("escenarios", nombre_archivo)
                shutil.copy2(filename, destino)

                # Actualizar referencia al archivo
                self.excel_file = nombre_archivo

                # Mostrar el panel de la consola y el nombre del archivo
                self.frame_consola.grid()
                self.consola.delete("1.0", tk.END)  # Limpiar consola
                self.consola.insert("end", f"Archivo Excel actual: {nombre_archivo}\n")
                self.consola.insert("end", f"Ruta de importación: {nombre_archivo}")
                self.consola.see("end")

                # Actualizar estado de los botones después de importar
                self.btn_importar.configure(state="disabled")
                self.btn_optimizar.configure(state="normal")
                self.btn_reiniciar.configure(state="normal")

                # Actualizar el estado en el panel inferior
                for seccion in self.secciones.values():
                    seccion.actualizar_estado(f"Archivo cargado {nombre_archivo}")

            except Exception as e:
                self.consola.insert("end", f"\nError al importar el archivo: {str(e)}")
                self.consola.see("end")
                # Actualizar el estado en caso de error
                for seccion in self.secciones.values():
                    seccion.actualizar_estado("Error al cargar el archivo")

    def reiniciar_sistema(self):
        """Reinicia el sistema a su estado inicial"""
        # Limpiar datos de simulación
        self.simulation_data = None
        self.simulation_running = False
        self.excel_file = None

        # Limpiar consola
        self.consola.delete("1.0", tk.END)
        self.consola.insert(
            "end", "Sistema reiniciado. Por favor, importe un nuevo archivo Excel."
        )
        self.consola.see("end")

        # Deshabilitar secciones
        self.boton_seccion1.configure(state="disabled")
        self.boton_seccion2.configure(state="disabled")
        self.boton_seccion3.configure(state="disabled")

        # Volver a la sección de inicio
        self.mostrar_inicio()

        # Restablecer estado inicial de los botones
        self.btn_importar.configure(state="normal")
        self.btn_optimizar.configure(
            text="Iniciar Optimización",
            fg_color=self.color_activo,
            text_color="white",
            state="disabled",
        )
        self.btn_reiniciar.configure(state="disabled")

        # Restablecer el estado en el panel inferior (sin mostrar archivo)
        for seccion in self.secciones.values():
            seccion.actualizar_estado("Selecciona un archivo Excel")

        # Limpiar gráficas
        if hasattr(self, "secciones"):
            for seccion in ["seccion1", "seccion2", "seccion3"]:
                if seccion in self.secciones:
                    for widget in self.secciones[
                        seccion
                    ].panel_contenido.winfo_children():
                        widget.destroy()

                    # Recrear mensaje inicial
                    frame_grafica = ctk.CTkFrame(
                        self.secciones[seccion].panel_contenido, fg_color="white"
                    )
                    frame_grafica.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

                    ctk.CTkLabel(
                        frame_grafica,
                        text="Ejecute la optimización para ver los resultados",
                        font=ctk.CTkFont(size=16),
                        text_color="gray60",
                    ).place(relx=0.5, rely=0.5, anchor="center")


def resource_path(relative_path):
    """Obtiene la ruta absoluta a un recurso,funcionando tanto en desarrollo como en el ejecutable PyInstaller."""
    try:
        # PyInstaller crea una carpeta temporal y la añade a _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Si no está en un ejecutable PyInstaller, usa la ruta base del script
        base_path = os.path.abspath(".")

    # Combina la ruta base con la ruta relativa del recurso
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    app = Aplicacion()
    app.mainloop()
