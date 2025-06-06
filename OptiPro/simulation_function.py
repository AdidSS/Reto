# simulation_function.py - Sistema de simulación logística que optimiza la distribución de plantas, gestionando inventario, rutas, pedidos y demanda para maximizar la eficiencia operativa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def cargar_parametros_desde_excel(ruta_excel):
    """
    Carga todos los parámetros desde un archivo Excel con múltiples hojas

    Args:
        ruta_excel: Ruta al archivo Excel con las hojas de parámetros

    Returns:
        params: Diccionario con todos los parámetros cargados
    """
    params = {}

    # 1. Cargar parámetros generales desde la hoja "parametros"
    df_params = pd.read_excel(ruta_excel, sheet_name="parametros")
    params["parametros_generales"] = {row[0]: row[1] for _, row in df_params.iterrows()}

    # Convertir tipos de datos según el tipo esperado
    for key in [
        "almacenamiento",
        "costo_plantacion",
        "dias_anticipacion",
        "min_dias_aclimatacion",
        "max_dias_aclimatacion",
        "dias_simulacion",
        "velocidad_camioneta",
        "tiempo_maximo",
        "tiempo_carga",
        "carga_maxima",
        "capacidad_maxima_transporte",
        "costo_transporte",
    ]:
        if key in params["parametros_generales"]:
            if key in [
                "dias_anticipacion",
                "min_dias_aclimatacion",
                "max_dias_aclimatacion",
                "dias_simulacion",
            ]:
                params["parametros_generales"][key] = int(
                    params["parametros_generales"][key]
                )
            else:
                params["parametros_generales"][key] = float(
                    params["parametros_generales"][key]
                )

    # 2. Cargar hectáreas por polígono desde la hoja "hectareas_poligono"
    df_hectareas = pd.read_excel(ruta_excel, sheet_name="hectareas_poligono")
    params["hectareas_poligono"] = dict(
        zip(df_hectareas["poligono"], df_hectareas["hectareas"])
    )

    # 3. Cargar demanda por especie desde la hoja "demanda_especies"
    df_demanda = pd.read_excel(ruta_excel, sheet_name="demanda_especies")
    params["demanda_especies_por_hectarea"] = dict(
        zip(df_demanda["especie"], df_demanda["demanda_por_hectarea"])
    )

    # 4. Cargar costos de proveedores desde la hoja "costos_proveedores"
    df_costos = pd.read_excel(ruta_excel, sheet_name="costos_proveedores")

    # Extraer nombres de proveedores (de las columnas excluyendo la primera)
    params["nombres_proveedores"] = df_costos.columns[1:].tolist()

    # Extraer nombres de especies (de la primera columna)
    params["nombres_especies"] = df_costos.iloc[:, 0].tolist()

    # Extraer matriz de costos (convertir a float, eliminando la primera columna)
    params["matriz_costos"] = df_costos.iloc[:, 1:].values.astype(float)

    # 5. Cargar distancias entre polígonos desde la hoja "distancias_poligonos"
    df_distancias = pd.read_excel(
        ruta_excel, sheet_name="distancias_poligonos", header=None
    )
    params["distancias_poligonos"] = df_distancias.values

    # Información adicional
    cprint(f"Parámetros cargados correctamente del archivo Excel: {ruta_excel}")
    cprint(f"Se cargaron {len(params['hectareas_poligono'])} polígonos")
    cprint(
        f"Se cargaron {len(params['nombres_especies'])} especies y {len(params['nombres_proveedores'])} proveedores"
    )

    return params


def obtener_parametros_simulacion(params):
    """Extrae e imprime los parámetros generales de simulación y retorna todos los valores necesarios."""
    parametros_generales = params["parametros_generales"]
    almacenamiento = parametros_generales.get("almacenamiento", 400)
    espacio_por_planta = parametros_generales.get("espacio_por_planta", 0.04)
    capacidad_maxima_plantas = int(almacenamiento / espacio_por_planta)
    costo_plantacion = parametros_generales.get("costo_plantacion", 20)
    dias_anticipacion = parametros_generales.get("dias_anticipacion", 1)
    min_dias_aclimatacion = parametros_generales.get("min_dias_aclimatacion", 3)
    max_dias_aclimatacion = parametros_generales.get("max_dias_aclimatacion", 7)
    dias_simulacion = parametros_generales.get("dias_simulacion", 10)
    velocidad_camioneta = parametros_generales.get("velocidad_camioneta", 40)
    tiempo_maximo = parametros_generales.get("tiempo_maximo", 6)
    tiempo_carga = parametros_generales.get("tiempo_carga", 0.5)
    carga_maxima = parametros_generales.get("carga_maxima", 524)
    capacidad_maxima_transporte = parametros_generales.get(
        "capacidad_maxima_transporte", 8000
    )
    costo_transporte = parametros_generales.get("costo_transporte", 4500)
    start_polygon = int(parametros_generales.get("start_polygon", 6))

    cprint("\n--- PARÁMETROS DE SIMULACIÓN (CARGADOS DE EXCEL) ---")
    cprint(f"Días de simulación: {dias_simulacion}")
    cprint(f"Capacidad máxima de transporte: {capacidad_maxima_transporte}")
    cprint(f"Días de anticipación para pedidos: {dias_anticipacion}")
    cprint(f"Días mínimos de aclimatación: {min_dias_aclimatacion}")
    cprint(f"Capacidad máxima de plantas en inventario: {capacidad_maxima_plantas}")
    cprint(f"Días máximos de aclimatación: {max_dias_aclimatacion}")
    cprint(f"Capacidad de carga por ruta: {carga_maxima}")
    cprint(f"Tiempo máximo por ruta: {tiempo_maximo} horas")

    return (
        almacenamiento,
        espacio_por_planta,
        capacidad_maxima_plantas,
        costo_plantacion,
        dias_anticipacion,
        min_dias_aclimatacion,
        max_dias_aclimatacion,
        dias_simulacion,
        velocidad_camioneta,
        tiempo_maximo,
        tiempo_carga,
        carga_maxima,
        capacidad_maxima_transporte,
        costo_transporte,
        start_polygon,
    )


def calculate_total_demand(hectareas_poligono, demanda_especies_por_hectarea):
    """Calculate the total demand for each species across all polygons"""
    # Create dictionary to store demand per species per polygon
    polygon_species_demand = {}

    # Calculate demand for each polygon based on hectares and species requirements
    for polygon_id, hectares in hectareas_poligono.items():
        polygon_species_demand[polygon_id] = {}
        for species_id, demand_per_hectare in demanda_especies_por_hectarea.items():
            # Calculate demand for this species in this polygon
            polygon_species_demand[polygon_id][species_id] = round(
                hectares * demand_per_hectare
            )

    # Calculate total demand per species across all polygons
    total_species_demand = {
        species_id: 0 for species_id in demanda_especies_por_hectarea.keys()
    }
    for polygon_id, species_demands in polygon_species_demand.items():
        for species_id, demand in species_demands.items():
            total_species_demand[species_id] += demand

    return polygon_species_demand, total_species_demand


class InventoryManager:
    """Class to manage plant inventory, acclimation and orders"""

    def __init__(
        self,
        demanda_especies_por_hectarea,
        min_dias_aclimatacion=3,
        max_dias_aclimatacion=7,
        capacidad_maxima_plantas=None,
    ):
        self.inventory = {}  # {species_id: [(quantity, days_in_inventory), ...]}
        self.pending_orders = {}  # {delivery_day: {species_id: quantity}}
        self.current_day = 0
        self.total_ordered = {
            species_id: 0 for species_id in demanda_especies_por_hectarea.keys()
        }
        self.history = []  # Daily inventory snapshots
        self.min_dias_aclimatacion = min_dias_aclimatacion
        self.max_dias_aclimatacion = max_dias_aclimatacion
        self.capacidad_maxima_plantas = (
            capacidad_maxima_plantas  # NUEVO: capacidad máxima de plantas
        )

    def place_order(self, orders, delivery_day):
        """Place orders with suppliers for delivery on specified day"""
        if delivery_day not in self.pending_orders:
            self.pending_orders[delivery_day] = {}

        for species_id, quantity in orders.items():
            if species_id in self.pending_orders[delivery_day]:
                self.pending_orders[delivery_day][species_id] += quantity
            else:
                self.pending_orders[delivery_day][species_id] = quantity

            # Track total ordered per species
            if species_id in self.total_ordered:
                self.total_ordered[species_id] += quantity

    def receive_deliveries(self, day):
        """Receive any deliveries scheduled for today"""
        if day in self.pending_orders:
            # Calcular cuántas plantas hay actualmente en inventario
            total_inventario = sum(
                qty for items in self.inventory.values() for qty, _ in items
            )
            # Calcular cuántas plantas se van a recibir
            total_a_recibir = sum(self.pending_orders[day].values())
            # Verificar si excede la capacidad máxima
            if (
                self.capacidad_maxima_plantas is not None
                and (total_inventario + total_a_recibir) > self.capacidad_maxima_plantas
            ):
                # Ajustar la cantidad a recibir para no exceder la capacidad
                espacio_disponible = self.capacidad_maxima_plantas - total_inventario
                if espacio_disponible <= 0:
                    cprint("Advertencia: Inventario lleno, no se reciben plantas hoy.")
                    del self.pending_orders[day]
                    return
                # Repartir el espacio disponible proporcionalmente
                total_pedidos = sum(self.pending_orders[day].values())
                for species_id in list(self.pending_orders[day].keys()):
                    cantidad_original = self.pending_orders[day][species_id]
                    cantidad_ajustada = int(
                        round(cantidad_original * espacio_disponible / total_pedidos)
                    )
                    # Asegurar al menos 0
                    self.pending_orders[day][species_id] = max(0, cantidad_ajustada)
                cprint(
                    f"Advertencia: Se ajustaron las entregas para no exceder la capacidad máxima de inventario ({self.capacidad_maxima_plantas} plantas)."
                )
            for species_id, quantity in self.pending_orders[day].items():
                if quantity <= 0:
                    continue
                if species_id not in self.inventory:
                    self.inventory[species_id] = []
                # Add to inventory with 0 days acclimation
                self.inventory[species_id].append((quantity, 0))
            # Clear processed orders
            del self.pending_orders[day]

    def update_inventory(self):
        """Age inventory by one day and take daily snapshot"""
        # Take snapshot before updating
        snapshot = self._get_inventory_snapshot()
        self.history.append(snapshot)

        # Update age of all plants
        for species_id in self.inventory:
            self.inventory[species_id] = [
                (qty, days + 1) for qty, days in self.inventory[species_id]
            ]

    def _get_inventory_snapshot(self):
        """Create a snapshot of current inventory state"""
        snapshot = {"total": {}, "available": {}, "by_age": {}}

        for species_id, items in self.inventory.items():
            # Total quantity by species
            total_qty = sum(qty for qty, _ in items)
            snapshot["total"][species_id] = total_qty

            # Available quantity (3-7 days)
            avail_qty = sum(
                qty
                for qty, days in items
                if self.min_dias_aclimatacion <= days <= self.max_dias_aclimatacion
            )
            snapshot["available"][species_id] = avail_qty

            # Group by days
            by_age = {}
            for qty, days in items:
                if days not in by_age:
                    by_age[days] = 0
                by_age[days] += qty
            snapshot["by_age"][species_id] = by_age

        return snapshot

    def get_available_inventory(self):
        """Get inventory items available for transport (3-7 days old)"""
        available = {}
        for species_id, items in self.inventory.items():
            available_qty = sum(
                qty
                for qty, days in items
                if self.min_dias_aclimatacion <= days <= self.max_dias_aclimatacion
            )
            if available_qty > 0:
                available[species_id] = available_qty
        return available

    def get_inventory_summary(self):
        """Get a summary of current inventory"""
        summary = {}
        for species_id, items in self.inventory.items():
            total = sum(qty for qty, _ in items)
            available = sum(
                qty
                for qty, days in items
                if self.min_dias_aclimatacion <= days <= self.max_dias_aclimatacion
            )
            too_young = sum(
                qty for qty, days in items if days < self.min_dias_aclimatacion
            )
            too_old = sum(
                qty for qty, days in items if days > self.max_dias_aclimatacion
            )

            summary[species_id] = {
                "total": total,
                "available": available,
                "too_young": too_young,
                "too_old": too_old,
            }
        return summary

    def remove_from_inventory(self, species_distribution):
        """Remove distributed items from inventory, prioritizing oldest items"""
        for species_id, qty_needed in species_distribution.items():
            if species_id not in self.inventory or qty_needed <= 0:
                continue

            # Sort by age (oldest first)
            self.inventory[species_id].sort(key=lambda x: x[1], reverse=True)

            qty_remaining = qty_needed
            new_inventory = []

            for qty, days in self.inventory[species_id]:
                if (
                    self.min_dias_aclimatacion <= days <= self.max_dias_aclimatacion
                    and qty_remaining > 0
                ):
                    # This batch is available for distribution
                    if qty <= qty_remaining:
                        # Use entire batch
                        qty_remaining -= qty
                    else:
                        # Use part of batch
                        new_inventory.append((qty - qty_remaining, days))
                        qty_remaining = 0
                else:
                    # Either not available or no more needed
                    new_inventory.append((qty, days))

            self.inventory[species_id] = new_inventory


def calculate_route_time(route, dist_poligonos_hrs):
    if len(route) <= 1:
        return 0

    total_time = 0
    # Sumar los tiempos de viaje entre polígonos consecutivos
    for i in range(len(route) - 1):
        total_time += dist_poligonos_hrs[route[i], route[i + 1]]

    # Añadir tiempos de carga/descarga (0.5 hrs por polígono visitado) más 0.5 extra
    # Se resta 2 porque no contamos como paradas el polígono inicial y final si son el mismo
    total_time += 0.5 * (len(route) - 2) + 0.5

    return total_time


def generar_mejores_rutas_greedy(
    current_demand,
    available_inventory,
    dist_poligonos_hrs,
    START_POLYGON,
    tiempo_maximo,
    carga_maxima,
    max_poligonos=10,
):
    """
    Genera rutas greedy priorizando la máxima entrega de plantas (todas las especies)
    y usando el tiempo de ruta como criterio secundario.
    """
    NUM_POLYGONS = dist_poligonos_hrs.shape[0]
    poligonos_restantes = [
        i for i in range(NUM_POLYGONS) if i != START_POLYGON and i in current_demand
    ]
    rutas = []

    while poligonos_restantes:
        mejor_ruta = None
        mejor_entrega = 0
        mejor_tiempo = None

        # Probar rutas cortas de 1 hasta max_poligonos
        for tam in range(1, min(max_poligonos, len(poligonos_restantes)) + 1):
            # Probar todas las combinaciones posibles de tamaño tam (puedes optimizar usando heurística)
            for seleccion in [
                poligonos_restantes[i : i + tam]
                for i in range(len(poligonos_restantes) - tam + 1)
            ]:
                ruta = [START_POLYGON] + seleccion + [START_POLYGON]
                tiempo = calculate_route_time(ruta, dist_poligonos_hrs)
                if tiempo > tiempo_maximo:
                    continue

                # Calcular cuántas plantas se podrían entregar en esta ruta
                entrega_total = 0
                carga_restante = carga_maxima
                for poligono in seleccion:
                    if poligono not in current_demand:
                        continue
                    for especie, demanda in current_demand[poligono].items():
                        disponible = available_inventory.get(especie, 0)
                        a_entregar = min(demanda, disponible, carga_restante)
                        entrega_total += a_entregar
                        carga_restante -= a_entregar
                        if carga_restante <= 0:
                            break
                    if carga_restante <= 0:
                        break

                # Guardar la mejor ruta (más entrega, menor tiempo)
                if entrega_total > mejor_entrega or (
                    entrega_total == mejor_entrega
                    and (mejor_tiempo is None or tiempo < mejor_tiempo)
                ):
                    mejor_ruta = ruta
                    mejor_entrega = entrega_total
                    mejor_tiempo = tiempo

        if mejor_ruta is None or mejor_entrega == 0:
            # No se puede hacer más rutas útiles
            break

        rutas.append((mejor_ruta, mejor_tiempo, mejor_entrega))

        # Marcar polígonos visitados como atendidos (puedes ajustar según si quieres permitir visitas múltiples)
        for poligono in mejor_ruta[1:-1]:
            if poligono in poligonos_restantes:
                poligonos_restantes.remove(poligono)

    return rutas


def calcular_max_plantas_repartibles(
    tiempo_maximo,
    tiempo_carga,
    carga_maxima,
    min_dias_aclimatacion,
    max_dias_aclimatacion,
):
    """
    Calcula el máximo de plantas que se pueden repartir en el rango de días útiles.
    """
    # Suponiendo viajes cortos: tiempo mínimo por viaje = tiempo_carga*2 (salida y regreso)
    tiempo_min_viaje = tiempo_carga * 2
    max_viajes_dia = int(tiempo_maximo // tiempo_min_viaje)
    dias_utiles = max_dias_aclimatacion - min_dias_aclimatacion + 1
    return max_viajes_dia * carga_maxima * dias_utiles


def plantas_en_rango_util(
    inventory, species_id, day, min_dias_aclimatacion, max_dias_aclimatacion
):
    # Plantas en inventario que estarán en rango útil en los próximos días
    en_rango = sum(
        qty
        for qty, days in inventory.inventory.get(species_id, [])
        if min_dias_aclimatacion <= days <= max_dias_aclimatacion
    )
    # Pedidos en tránsito que llegarán en el rango útil
    en_transito = 0
    for entrega_dia, pedidos in inventory.pending_orders.items():
        dias_hasta_entrega = entrega_dia - day
        if 0 <= dias_hasta_entrega <= (max_dias_aclimatacion - min_dias_aclimatacion):
            en_transito += pedidos.get(species_id, 0)
    return en_rango + en_transito


def place_orders_with_suppliers(
    inventory,
    remaining_demand,
    matriz_costos,
    costo_transporte,
    capacidad_maxima_transporte,
    day,
    delivery_day,
    nombres_especies,
    nombres_proveedores,
    dias_anticipacion=1,
    min_dias_aclimatacion=3,
    max_dias_aclimatacion=7,
    tiempo_maximo=6,
    tiempo_carga=0.5,
    carga_maxima=524,
    current_demand=None,
):
    """
    Heurística: Solo pedir lo que se puede repartir en el rango de días útiles.
    """
    cprint(f"\n> Calculando pedidos para entrega en día {delivery_day}")

    max_plantas_rango = calcular_max_plantas_repartibles(
        tiempo_maximo,
        tiempo_carga,
        carga_maxima,
        min_dias_aclimatacion,
        max_dias_aclimatacion,
    )

    # Calcular demanda futura en el rango de días útiles
    demanda_futura = {}
    if current_demand is not None:
        for species_id in remaining_demand:
            demanda_futura[species_id] = 0
        for polygon, species_demands in current_demand.items():
            for species_id, qty in species_demands.items():
                if species_id in demanda_futura:
                    demanda_futura[species_id] += qty
    else:
        demanda_futura = remaining_demand.copy()

    # Limitar pedido por especie al máximo que se puede repartir en el rango
    orders_by_species = {}
    total_cost = 0
    total_ordered = 0

    for species_id, demand in demanda_futura.items():
        # Plantas que estarán disponibles en el rango útil (inventario + pedidos en tránsito)
        total_en_rango = plantas_en_rango_util(
            inventory, species_id, day, min_dias_aclimatacion, max_dias_aclimatacion
        )
        pedido = max(0, min(demand, max_plantas_rango) - total_en_rango)
        if pedido <= 0:
            continue
        species_index = int(species_id) - 1
        if species_index < 0 or species_index >= len(matriz_costos):
            cprint(f"Error: Species ID {species_id} out of range in cost matrix")
            continue
        provider_costs = matriz_costos[species_index]
        min_cost_idx = np.argmin(provider_costs)
        min_cost = provider_costs[min_cost_idx]
        if species_id not in orders_by_species:
            orders_by_species[species_id] = {}
        provider_id = min_cost_idx + 1
        orders_by_species[species_id][provider_id] = pedido
        total_cost += pedido * min_cost + costo_transporte
        total_ordered += pedido
        cprint(
            f"   Ordenando {pedido} plantas de especie '{nombres_especies[species_index]}' "
            f"a proveedor '{nombres_proveedores[min_cost_idx]}' "
            f"a ${min_cost:.2f}/planta (máx repartible en rango: {max_plantas_rango})"
        )

    # Registrar pedidos en el inventario
    for species_id, providers in orders_by_species.items():
        total_qty = sum(providers.values())
        inventory.place_order({species_id: total_qty}, delivery_day)

    cprint(
        f"   Costo total de pedido: ${total_cost:.2f} (Total plantas: {total_ordered})"
    )
    return orders_by_species


def distribute_plants_to_routes(inventory, current_demand, routes, carga_maxima):
    """
    Distribuye plantas priorizando especies con más plantas próximas a volverse viejas.
    """
    available_inventory = inventory.get_available_inventory()
    if not available_inventory:
        cprint("   No hay inventario disponible para distribuir hoy")
        return {}, {}

    # 1. Calcular cuántas plantas están próximas a ser viejas por especie
    species_oldness = {}
    for species_id, items in inventory.inventory.items():
        # Prioriza plantas con días = max_dias_aclimatacion-2 o mayores (pero >= min)
        count = sum(
            qty
            for qty, days in items
            if days
            >= max(inventory.max_dias_aclimatacion - 2, inventory.min_dias_aclimatacion)
        )
        species_oldness[species_id] = count

    # 2. Ordenar especies por mayor cantidad de plantas próximas a ser viejas
    sorted_species = sorted(
        available_inventory.keys(),
        key=lambda s: species_oldness.get(s, 0),
        reverse=True,
    )

    distribution_plan = {}
    route_loads = {}

    for route_idx in range(len(routes)):
        distribution_plan[route_idx] = {}
        route_loads[route_idx] = 0
        route_path = routes[route_idx][0]
        for polygon in route_path[1:-1]:
            if polygon in current_demand:
                distribution_plan[route_idx][polygon] = {}
                # Usar el orden priorizado de especies
                for species_id in sorted_species:
                    demand = current_demand[polygon].get(species_id, 0)
                    if demand > 0 and species_id in available_inventory:
                        available = available_inventory[species_id]
                        to_deliver = min(
                            demand, available, carga_maxima - route_loads[route_idx]
                        )
                        if to_deliver > 0:
                            distribution_plan[route_idx][polygon][
                                species_id
                            ] = to_deliver
                            available_inventory[species_id] -= to_deliver
                            route_loads[route_idx] += to_deliver
                            if available_inventory[species_id] <= 0:
                                del available_inventory[species_id]

    # Remove plants from inventory
    total_distribution = {}
    for route_idx, polygon_dist in distribution_plan.items():
        for polygon, species_dist in polygon_dist.items():
            for species_id, qty in species_dist.items():
                if species_id not in total_distribution:
                    total_distribution[species_id] = 0
                total_distribution[species_id] += qty

    inventory.remove_from_inventory(total_distribution)

    # Report distribution
    if sum(route_loads.values()) > 0:
        cprint("\n> Plan de distribución para hoy:")
        for route_idx, load in route_loads.items():
            if load > 0:
                cprint(
                    f"   Ruta {route_idx + 1}: {load} plantas (Carga máxima: {carga_maxima})"
                )
    else:
        cprint("   No se distribuyeron plantas hoy")

    return distribution_plan, route_loads


def update_demand_after_distribution(current_demand, distribution_plan):
    """Update remaining demand after plants have been distributed"""
    updated_demand = {}

    # Make a deep copy of current demand
    for polygon, species_demands in current_demand.items():
        updated_demand[polygon] = {
            species: qty for species, qty in species_demands.items()
        }

    # Subtract distributed plants
    for route_idx, polygon_dist in distribution_plan.items():
        for polygon, species_dist in polygon_dist.items():
            if polygon in updated_demand:
                for species_id, qty in species_dist.items():
                    if species_id in updated_demand[polygon]:
                        updated_demand[polygon][species_id] -= qty
                        # Ensure non-negative demand
                        updated_demand[polygon][species_id] = max(
                            0, updated_demand[polygon][species_id]
                        )

    return updated_demand


def calculate_demand_coverage(total_species_demand, current_demand):
    """Calculate percentage of demand that has been covered"""
    coverage = {}

    # Calculate remaining total demand by species
    remaining_demand = {species_id: 0 for species_id in total_species_demand.keys()}
    for polygon, species_demands in current_demand.items():
        for species_id, qty in species_demands.items():
            if species_id in remaining_demand:
                remaining_demand[species_id] += qty

    # Calculate coverage percentage
    for species_id, total in total_species_demand.items():
        if total > 0:
            coverage[species_id] = 1.0 - (remaining_demand[species_id] / total)
        else:
            coverage[species_id] = 1.0  # No demand means 100% coverage

    return coverage


def generar_reporte_diario(
    day,
    current_orders,
    today_route,
    distribution_plan,
    route_loads,
    inventory,
    current_demand,
    coverage,
    nombres_especies,
):
    """Generate a daily report of inventory, orders, and distributions"""
    cprint("\n> Resumen diario:")

    # Inventory summary
    inv_summary = inventory.get_inventory_summary()
    cprint("   Inventario actual:")
    for species_id, info in inv_summary.items():
        species_idx = int(species_id) - 1
        species_name = (
            nombres_especies[species_idx]
            if species_idx < len(nombres_especies)
            else f"Especie {species_id}"
        )

        cprint(
            f"      {species_name}: {info['total']} plantas totales "
            f"({info['available']} disponibles, {info['too_young']} muy jóvenes, "
            f"{info['too_old']} muy viejas)"
        )

    # Demand coverage
    cprint("   Cobertura de demanda:")
    avg_coverage = 0
    for species_id, pct in coverage.items():
        species_idx = int(species_id) - 1
        species_name = (
            nombres_especies[species_idx]
            if species_idx < len(nombres_especies)
            else f"Especie {species_id}"
        )
        cprint(f"      {species_name}: {pct:.2%}")
        avg_coverage += pct

    if coverage:
        avg_coverage /= len(coverage)
        cprint(f"   Cobertura promedio: {avg_coverage:.2%}")


def visualizar_cobertura_demanda(daily_demand_coverage, nombres_especies):
    """Visualize demand coverage over time"""
    if not daily_demand_coverage:
        return

    fig = plt.figure(figsize=(12, 6))

    # Extract coverage by species and day
    days = range(1, len(daily_demand_coverage) + 1)

    # Get all species IDs from coverage data
    all_species = set()
    for day_coverage in daily_demand_coverage:
        all_species.update(day_coverage.keys())

    # Plot coverage for each species
    for species_id in sorted(all_species):
        coverage_values = [
            day_coverage.get(species_id, 0) * 100
            for day_coverage in daily_demand_coverage
        ]
        species_idx = int(species_id) - 1
        species_name = (
            nombres_especies[species_idx]
            if species_idx < len(nombres_especies)
            else f"Especie {species_id}"
        )
        plt.plot(days, coverage_values, marker="o", label=species_name)

    plt.title("Cobertura de Demanda a lo Largo del Tiempo")
    plt.xlabel("Día de Simulación")
    plt.ylabel("Cobertura (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig


def visualizar_inventario(inventory_history, nombres_especies):
    """Visualize inventory levels over time"""
    if not inventory_history:
        return

    fig = plt.figure(figsize=(12, 6))

    # Extract available inventory by species and day
    days = range(1, len(inventory_history) + 1)

    # Get all species from inventory history
    all_species = set()
    for day_inventory in inventory_history:
        if "available" in day_inventory:
            all_species.update(day_inventory["available"].keys())

    # Plot available inventory for each species
    for species_id in sorted(all_species):
        inventory_values = [
            day_inventory.get("available", {}).get(species_id, 0)
            for day_inventory in inventory_history
        ]

        species_idx = int(species_id) - 1
        species_name = (
            nombres_especies[species_idx]
            if species_idx < len(nombres_especies)
            else f"Especie {species_id}"
        )
        plt.plot(days, inventory_values, marker="s", label=species_name)

    plt.title("Inventario Disponible por Día")
    plt.xlabel("Día de Simulación")
    plt.ylabel("Plantas Disponibles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    return fig


def run_simulation(
    dias_simulacion,
    dias_anticipacion,
    min_dias_aclimatacion,
    max_dias_aclimatacion,
    hectareas_poligono,
    demanda_especies_por_hectarea,
    capacidad_maxima_transporte,
    costo_transporte,
    carga_maxima,
    nombres_especies,
    nombres_proveedores,
    matriz_costos,
    dist_poligonos_hrs,
    NUM_POLYGONS,
    START_POLYGON,
    tiempo_maximo,
    capacidad_maxima_plantas,
    tiempo_carga,
):
    """Run the full simulation for the specified number of days"""
    # Calculate initial demands
    polygon_species_demand, total_species_demand = calculate_total_demand(
        hectareas_poligono=hectareas_poligono,
        demanda_especies_por_hectarea=demanda_especies_por_hectarea,
    )

    # Convert total demand to list format for reporting
    initial_demand_list = [
        total_species_demand.get(i, 0) for i in range(1, len(nombres_especies) + 1)
    ]

    cprint(f"\n--- DEMANDA INICIAL ---")
    for i, demand in enumerate(initial_demand_list):
        cprint(f"{nombres_especies[i]}: {demand}")

    # Create inventory manager
    inventory = InventoryManager(
        demanda_especies_por_hectarea,
        min_dias_aclimatacion=min_dias_aclimatacion,
        max_dias_aclimatacion=max_dias_aclimatacion,
        capacidad_maxima_plantas=capacidad_maxima_plantas,
    )

    # Daily tracking of orders, routes, and demand fulfillment
    daily_orders = {}
    daily_routes = {}
    daily_distributions = {}
    daily_demand_coverage = []

    # Deep copy of initial demand for tracking
    current_demand = {
        polygon: {species: qty for species, qty in species_demand.items()}
        for polygon, species_demand in polygon_species_demand.items()
    }

    # Calculate all routes at the beginning of simulation
    # all_routes = generar_mejores_rutas_greedy(current_demand, available_inventory, dist_poligonos_hrs, START_POLYGON, tiempo_maximo, carga_maxima)

    # cprint(f"\n--- RUTAS CALCULADAS ---")
    # cprint(f"Se han calculado {len(all_routes)} rutas")

    # Lista de riguras para rutas
    fig_rutas = []

    for day in range(dias_simulacion):
        cprint(f"\n{'='*50}")
        cprint(f"SIMULACIÓN DÍA {day+1}")
        cprint(f"{'='*50}")

        # 1. Receive any pending deliveries
        inventory.receive_deliveries(day)

        # 2. Verificar si hay plantas disponibles para repartir
        available_inventory = inventory.get_available_inventory()
        hay_plantas = any(qty > 0 for qty in available_inventory.values())

        if hay_plantas:
            all_routes = generar_mejores_rutas_greedy(
                current_demand,
                available_inventory,
                dist_poligonos_hrs,
                START_POLYGON,
                tiempo_maximo,
                carga_maxima,
            )
            if all_routes:
                rutas_del_dia = []
                tiempo_acumulado = 0
                for ruta, tiempo, _ in all_routes:
                    if tiempo_acumulado + tiempo <= tiempo_maximo:
                        rutas_del_dia.append((ruta, tiempo, _))
                        tiempo_acumulado += tiempo
                    else:
                        break
                today_route = rutas_del_dia  # <-- ahora es una lista de rutas
                daily_routes[day] = today_route
                cprint(f"Rutas seleccionadas para hoy: {len(today_route)} rutas")

                # 4. Distribuir plantas en todas las rutas generadas hoy
                distribution_plan, route_loads = distribute_plants_to_routes(
                    inventory, current_demand, today_route, carga_maxima
                )
                daily_distributions[day] = distribution_plan

                # 5. Actualizar demanda restante
                current_demand = update_demand_after_distribution(
                    current_demand, distribution_plan
                )
            else:
                cprint(
                    "No se encontraron rutas válidas para hoy. No se realiza distribución."
                )
                today_route = []
                daily_routes[day] = today_route
                distribution_plan = {}
                route_loads = {}
        else:
            cprint("No hay plantas disponibles para repartir hoy. No se realiza ruta.")
            today_route = []
            daily_routes[day] = today_route
            distribution_plan = {}
            route_loads = {}

        # 6. Calculate demand coverage
        coverage = calculate_demand_coverage(total_species_demand, current_demand)
        daily_demand_coverage.append(coverage)

        # 7. Order plants for future delivery (estrategia para minimizar plantas viejas)
        current_orders = None
        if day < dias_simulacion - dias_anticipacion:
            delivery_day = day + dias_anticipacion
            # Calcular demanda restante
            remaining_demand = {}
            for polygon, species_demands in current_demand.items():
                for species_id, demand in species_demands.items():
                    if demand > 0:
                        if species_id in remaining_demand:
                            remaining_demand[species_id] += demand
                        else:
                            remaining_demand[species_id] = demand
            # Pedir solo lo necesario para cubrir demanda futura y no sobrepedir
            if remaining_demand:
                current_orders = place_orders_with_suppliers(
                    inventory,
                    remaining_demand,
                    matriz_costos,
                    costo_transporte,
                    capacidad_maxima_transporte,
                    day,
                    delivery_day,
                    nombres_especies,
                    nombres_proveedores,
                    dias_anticipacion=dias_anticipacion,
                    min_dias_aclimatacion=min_dias_aclimatacion,
                    max_dias_aclimatacion=max_dias_aclimatacion,
                    tiempo_maximo=tiempo_maximo,
                    tiempo_carga=tiempo_carga,
                    carga_maxima=carga_maxima,
                    current_demand=current_demand,
                )
                daily_orders[day] = (delivery_day, current_orders)

        # 8. Generate daily report
        generar_reporte_diario(
            day,
            current_orders,
            today_route,
            distribution_plan,
            route_loads,
            inventory,
            current_demand,
            coverage,
            nombres_especies,
        )

        # 9. Update inventory ages
        inventory.update_inventory()

        # 10. Visualize today's route distribution (solo si hubo ruta)
        if today_route:
            fig = visualizar_distribucion_rutas(
                today_route,
                distribution_plan,
                day,
                nombres_especies,
                NUM_POLYGONS,
                START_POLYGON,
                dist_poligonos_hrs,
            )
            fig_rutas.append(fig)

    # End of simulation visualizations
    cprint("\n--- RESUMEN FINAL DE SIMULACIÓN ---")
    cprint(f"Días simulados: {dias_simulacion}")

    # Visualize demand coverage over time
    fig_demanda = visualizar_cobertura_demanda(daily_demand_coverage, nombres_especies)

    # Visualize inventory history
    fig_inventario = visualizar_inventario(inventory.history, nombres_especies)

    # Return simulation data
    return {
        "daily_orders": daily_orders,
        "daily_routes": daily_routes,
        "daily_distributions": daily_distributions,
        "daily_demand_coverage": daily_demand_coverage,
        "final_demand": current_demand,
        "inventory_history": inventory.history,
        "initial_demand_list": initial_demand_list,
        "fig_rutas": fig_rutas,
        "fig_demanda": fig_demanda,
        "fig_inventario": fig_inventario,
    }


def visualizar_distribucion_rutas(
    routes,
    distribution_plan,
    day,
    nombres_especies,
    NUM_POLYGONS,
    START_POLYGON,
    dist_poligonos_hrs,
):
    """Visualize today's route distribution"""
    if not routes:
        return
        # Crear grafo base con todos los nodos
    G = nx.DiGraph()
    for j in range(NUM_POLYGONS):
        G.add_node(j)

    pos = nx.spring_layout(G, seed=42)
    fig = plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[START_POLYGON],
        node_color="red",
        node_size=700,
        label="Depósito",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if n != START_POLYGON],
        node_color="lightblue",
        node_size=500,
    )
    nx.draw_networkx_labels(G, pos)

    # Colores para las rutas
    colores = plt.cm.get_cmap("tab10", len(routes))

    # Dibujar cada ruta con un color distinto
    for idx, (route, time, _) in enumerate(routes):
        edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=[colores(idx)],
            width=3,
            arrowsize=20,
            label=f"Ruta {idx+1}",
        )

    plt.title(f"Distribución de Rutas - Día {day+1} ({len(routes)} rutas)")
    plt.axis("off")
    plt.legend([f"Ruta {i+1}" for i in range(len(routes))], loc="upper left")
    plt.tight_layout()
    # plt.show()

    return fig


def prepararacion(archivo):
    # Cargar todos los parámetros desde archivo Excel
    archivo_ruta = f"escenarios/{archivo}"
    params = cargar_parametros_desde_excel(archivo_ruta)

    # Extraer parámetros generales
    # Con valores por defecto si no están presentes
    (
        almacenamiento,
        espacio_por_planta,
        capacidad_maxima_plantas,
        costo_plantacion,
        dias_anticipacion,
        min_dias_aclimatacion,
        max_dias_aclimatacion,
        dias_simulacion,
        velocidad_camioneta,
        tiempo_maximo,
        tiempo_carga,
        carga_maxima,
        capacidad_maxima_transporte,
        costo_transporte,
        start_polygon,
    ) = obtener_parametros_simulacion(params)

    # Extraer otros datos
    hectareas_poligono = params["hectareas_poligono"]
    demanda_especies_por_hectarea = params["demanda_especies_por_hectarea"]
    nombres_especies = params["nombres_especies"]
    nombres_proveedores = params["nombres_proveedores"]
    matriz_costos = params["matriz_costos"]

    # Procesar matriz de distancias
    distancias_poligonos = params["distancias_poligonos"]
    dist_poligonos_hrs = distancias_poligonos / velocidad_camioneta
    dist_poligonos_hrs = np.round(dist_poligonos_hrs, 2)

    # Determinar el número total de polígonos
    NUM_POLYGONS = min(
        dist_poligonos_hrs.shape[0], max(hectareas_poligono.keys()) + 1
    )  # +1 para incluir todos los polígonos
    START_POLYGON = start_polygon

    return (
        dias_simulacion,
        dias_anticipacion,
        min_dias_aclimatacion,
        max_dias_aclimatacion,
        hectareas_poligono,
        demanda_especies_por_hectarea,
        capacidad_maxima_plantas,
        tiempo_carga,
        carga_maxima,
        capacidad_maxima_transporte,
        costo_transporte,
        start_polygon,
        nombres_especies,
        nombres_proveedores,
        matriz_costos,
        dist_poligonos_hrs,
        NUM_POLYGONS,
        START_POLYGON,
        tiempo_maximo,
    )


# Función principal
def run_simulation_external(archivo):

    # Recupera parémtros de preparación
    (
        dias_simulacion,
        dias_anticipacion,
        min_dias_aclimatacion,
        max_dias_aclimatacion,
        hectareas_poligono,
        demanda_especies_por_hectarea,
        capacidad_maxima_plantas,
        tiempo_carga,
        carga_maxima,
        capacidad_maxima_transporte,
        costo_transporte,
        start_polygon,
        nombres_especies,
        nombres_proveedores,
        matriz_costos,
        dist_poligonos_hrs,
        num_polygons,
        start_polygon,
        tiempo_maximo,
    ) = prepararacion(archivo)

    # Ejecutar simulación con los parámetros cargados
    simulation_results = run_simulation(
        dias_simulacion=dias_simulacion,
        dias_anticipacion=dias_anticipacion,
        min_dias_aclimatacion=min_dias_aclimatacion,
        max_dias_aclimatacion=max_dias_aclimatacion,
        hectareas_poligono=hectareas_poligono,
        demanda_especies_por_hectarea=demanda_especies_por_hectarea,
        capacidad_maxima_transporte=capacidad_maxima_transporte,
        costo_transporte=costo_transporte,
        carga_maxima=carga_maxima,
        nombres_especies=nombres_especies,
        nombres_proveedores=nombres_proveedores,
        matriz_costos=matriz_costos,
        dist_poligonos_hrs=dist_poligonos_hrs,
        NUM_POLYGONS=num_polygons,
        START_POLYGON=start_polygon,
        tiempo_maximo=tiempo_maximo,
        capacidad_maxima_plantas=capacidad_maxima_plantas,
        tiempo_carga=tiempo_carga,
    )

    # Final summary messages
    final_coverage = simulation_results["daily_demand_coverage"][-1]
    avg_coverage = sum(final_coverage.values()) / len(final_coverage)

    cprint("\n--- RESULTADO FINAL ---")
    cprint(f"Cobertura de demanda promedio: {avg_coverage:.2%}")

    total_orders = sum(
        sum(providers.values())
        for day_info in simulation_results["daily_orders"].values()
        for _, orders in [day_info]
        for species, providers in orders.items()
    )

    cprint(f"Total de plantas ordenadas: {total_orders}")
    cprint("Simulación completada con éxito!")

    # Extract simulation data
    # daily_orders = simulation_results["daily_orders"]
    # daily_routes = simulation_results["daily_routes"]
    # daily_distributions = simulation_results["daily_distributions"]
    # daily_demand_coverage = simulation_results["daily_demand_coverage"]
    # final_demand = simulation_results["final_demand"]
    # inventory_history = simulation_results["inventory_history"]
    # initial_demand_list = simulation_results["initial_demand_list"]
    # fig_rutas = simulation_results["fig_rutas"]
    # fig_demanda = simulation_results["fig_demanda"]
    # fig_inventario = simulation_results["fig_inventario"]

    return simulation_results


# Lista global para acumular mensajes
console_messages = []


def cprint(*args, **kwargs):
    # Obtener timestamp actual
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convertir los argumentos a string como lo hace print()
    output = " ".join(str(arg) for arg in args)

    # Procesar kwargs como lo haría print()
    if "sep" in kwargs:
        output = kwargs["sep"].join(str(arg) for arg in args)
    if "end" in kwargs:
        output += kwargs["end"]
    else:
        output += "\n"

    # Agregar timestamp al mensaje
    output_with_timestamp = f"[{timestamp}] {output}"

    # Acumular el mensaje con timestamp
    console_messages.append(output_with_timestamp)
    # También imprimir en la consola real usando print() directamente
    print(output_with_timestamp)


if __name__ == "__main__":
    run_simulation_external("escenario-1.xlsx")
