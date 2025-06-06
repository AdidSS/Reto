# app_ui.py - Módulo que contiene los componentes y clases de la interfaz de usuario, incluyendo diálogos, secciones y estilos visuales
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk


# Clase que define la paleta de colores utilizada en toda la aplicación
class ColoresUI:
    """Clase que define la paleta de colores utilizada en toda la aplicación"""

    def __init__(self):
        # Colores para botones activos/inactivos y fondos
        self.activo = "#E65100"  # Naranja rojizo profundo para botón activo
        self.inactivo = "#1A2D5C"  # Color azul medio para botones inactivos
        self.hover = "#2853C9"  # Color hover para botones - azul brillante intermedio
        self.barra_colapso = "#F8F9FA"  # Color integrado con el fondo principal
        self.barra_hover = "#E9ECEF"  # Color hover para la barra de colapso
        self.fondo_barra_lateral = "#0A1128"  # Color azul oscuro para la barra lateral
        self.fondo_contenido = "#F8F9FA"  # Gris muy claro para el fondo principal
        self.texto_menu = "#FFFFFF"  # Color texto menú
        self.fondo_paneles = "#FFFFFF"  # Color blanco para paneles
        self.titulo = "#212529"  # Color texto título
        self.acerca = "#E65100"  # Color botón acerca, usando el mismo que el activo


# Ventana modal que muestra información sobre la aplicación y sus desarrolladores
class DialogoAcerca(ctk.CTkToplevel):
    """Ventana modal que muestra información sobre la aplicación y sus desarrolladores"""

    def __init__(self, parent):
        super().__init__(parent)

        # Configurar ventana de diálogo
        self.title("Acerca de OptiPro")
        self.geometry("750x500")

        # Hacer que la ventana sea modal
        self.transient(parent)
        self.grab_set()

        # Centrar en la pantalla
        self.update_idletasks()
        ancho = self.winfo_width()
        alto = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f"{ancho}x{alto}+{x}+{y}")

        # Configurar grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Título
        self.titulo = ctk.CTkLabel(
            self,
            text="OptiPro - Sistema de Optimización Logística",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.titulo.grid(row=0, column=0, padx=30, pady=(30, 10))

        # Versión
        self.version = ctk.CTkLabel(
            self, text="Versión 1.0.0", font=ctk.CTkFont(size=14), text_color="gray60"
        )
        self.version.grid(row=1, column=0, padx=30, pady=(0, 20))

        # Frame para el contenido
        self.frame_contenido = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.frame_contenido.grid(row=2, column=0, sticky="nsew", padx=30, pady=(0, 10))
        self.frame_contenido.grid_columnconfigure(0, weight=1)

        # Descripción
        self.descripcion = ctk.CTkTextbox(self.frame_contenido, height=230)
        self.descripcion.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.descripcion.insert(
            "1.0",
            """OptiPro es una aplicación de escritorio diseñada para optimizar la logística de distribución de plantas desde viveros hasta sitios de plantación. La aplicación proporciona una interfaz gráfica moderna y funcional para gestionar todo el proceso logístico.

Características principales:

• Gestión de Inventario
  - Control y seguimiento de plantas disponibles en vivero
  - Monitoreo de tiempos de aclimatación
  - Optimización del espacio en vivero
  - Control de capacidad máxima de almacenamiento

• Planificación de Rutas
  - Generación automática de rutas óptimas de distribución
  - Consideración de restricciones de tiempo y capacidad
  - Balanceo eficiente de cargas por ruta
  - Optimización de tiempos de entrega

• Análisis de Demanda
  - Visualización detallada de demanda por especie
  - Seguimiento en tiempo real de cobertura de demanda
  - Planificación inteligente de pedidos
  - Gestión de proveedores

• Optimización de Recursos
  - Minimización de costos de transporte
  - Maximización de eficiencia en entregas
  - Gestión óptima de tiempos de aclimatación
  - Control de capacidad de carga y tiempos máximos""",
        )
        self.descripcion.configure(state="disabled")

        # Autores
        self.etiqueta_autores = ctk.CTkLabel(
            self.frame_contenido,
            text="Desarrollado por:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.etiqueta_autores.grid(row=1, column=0, sticky="w", pady=(0, 0))

        self.autores = ctk.CTkLabel(
            self.frame_contenido,
            text="Leonardo De Regil Cárdenas, Eduardo Mitrani Krouham, Victor Adid Salgado Santana, Mario Alberto Landa Flores",
            justify="left",
        )
        self.autores.grid(row=2, column=0, sticky="w")

        # Botón cerrar
        self.boton_cerrar = ctk.CTkButton(self, text="Cerrar", command=self.destroy)
        self.boton_cerrar.grid(row=3, column=0, pady=20)


# Plantilla base para crear secciones de contenido con estructura y estilos consistentes
class SeccionContenido(ctk.CTkFrame):
    """Plantilla base para crear secciones de contenido con estructura y estilos consistentes"""

    def __init__(self, padre, titulo, color_fondo, color_titulo):
        super().__init__(padre, fg_color="transparent", corner_radius=0)

        # Guardar colores
        self.color_fondo = color_fondo
        self.color_titulo = color_titulo

        # Configurar grid principal
        self.grid(row=0, column=0, sticky="nsew")

        # Crear contenedor principal que tomará toda la altura
        self.contenedor_principal = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.contenedor_principal.grid(row=0, column=0, sticky="nsew")

        # Configurar grid del contenedor principal
        self.contenedor_principal.grid_rowconfigure(
            1, weight=1
        )  # Panel de contenido crece
        self.contenedor_principal.grid_rowconfigure(0, weight=0)  # Encabezado fijo
        self.contenedor_principal.grid_rowconfigure(2, weight=0)  # Panel inferior fijo
        self.contenedor_principal.grid_columnconfigure(0, weight=1)

        # Configurar grid de la página
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Panel Superior (altura fija: 80px)
        self.panel_superior = ctk.CTkFrame(
            self.contenedor_principal,
            fg_color="#FFFFFF",  # Panel blanco
            corner_radius=0,
            height=80,
            width=2000,
        )
        self.panel_superior.grid(row=0, column=0, sticky="new")
        self.panel_superior.grid_propagate(False)
        self.panel_superior.grid_columnconfigure(1, weight=1)

        # Marco para título y subtítulo
        self.marco_titulos = ctk.CTkFrame(
            self.panel_superior, fg_color="transparent", corner_radius=0
        )
        self.marco_titulos.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # Título
        self.etiqueta_titulo = ctk.CTkLabel(
            self.marco_titulos,
            text=titulo,
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.color_titulo,
        )
        self.etiqueta_titulo.grid(row=0, column=0, sticky="w")

        # Subtítulo
        self.etiqueta_subtitulo = ctk.CTkLabel(
            self.marco_titulos,
            text="Descripción de la página actual",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        self.etiqueta_subtitulo.grid(row=1, column=0, sticky="w", pady=(0, 5))

        # Área de notificaciones
        self.marco_notificaciones = ctk.CTkFrame(
            self.panel_superior, fg_color="transparent", corner_radius=0
        )
        self.marco_notificaciones.grid(row=0, column=1, padx=20, pady=10, sticky="e")

        self.etiqueta_notificaciones = ctk.CTkLabel(
            self.marco_notificaciones,
            text="",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        self.etiqueta_notificaciones.grid(row=0, column=0, sticky="e")

        # Línea separadora
        self.separador_superior = ctk.CTkFrame(
            self.panel_superior, height=2, fg_color="gray85"
        )
        self.separador_superior.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Panel de Contenido Principal - Simplificado a un solo contenedor
        self.panel_contenido = ctk.CTkFrame(
            self.contenedor_principal,
            fg_color="#FFFFFF",  # Panel blanco igual que los otros
            corner_radius=0,
            width=2000,
            height=600,  # Altura inicial, se ajustará con el grid
        )
        self.panel_contenido.grid(row=1, column=0, sticky="nsew")
        self.panel_contenido.grid_propagate(False)

        # Configurar el grid del panel de contenido para que ocupe todo el espacio
        self.panel_contenido.grid_columnconfigure(0, weight=1)
        self.panel_contenido.grid_rowconfigure(0, weight=1)

        # Panel Inferior (altura fija: 40px)
        self.panel_inferior = ctk.CTkFrame(
            self.contenedor_principal,
            fg_color="#FFFFFF",  # Panel blanco
            corner_radius=0,
            height=40,
            width=2000,
        )
        self.panel_inferior.grid(row=2, column=0, sticky="sew")
        self.panel_inferior.grid_propagate(False)
        self.panel_inferior.grid_columnconfigure(0, weight=1)

        # Línea separadora superior del panel inferior
        self.separador_inferior = ctk.CTkFrame(
            self.panel_inferior, height=1, fg_color="gray85"
        )
        self.separador_inferior.grid(row=0, column=0, sticky="new")

        # Área de estado en el panel inferior
        self.marco_estado = ctk.CTkFrame(
            self.panel_inferior, fg_color="transparent", corner_radius=0
        )
        self.marco_estado.grid(row=1, column=0, sticky="ew", padx=20)
        self.marco_estado.grid_columnconfigure(1, weight=1)

        # Etiqueta de estado
        self.etiqueta_estado = ctk.CTkLabel(
            self.marco_estado,
            text="Listo",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        self.etiqueta_estado.grid(row=0, column=0, sticky="w")

    def actualizar_estado(self, mensaje):
        """Actualiza el mensaje de estado en el panel inferior"""
        self.etiqueta_estado.configure(text=mensaje)
