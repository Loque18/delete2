from core.sim import Obstacle_c, Obstacle_wall, crear_paredes_v2

def simple_center_obstacle(arena_size=200, obstacle_radius=14.0):
    return [
        Obstacle_c(
            x_prop=arena_size / 2,
            y_prop=arena_size / 2,
            arena_size=int(arena_size),
            radius=obstacle_radius,
        )
    ]

def scene2(arena_size=200, wall_thickness=2):
    margin = wall_thickness / 2

    room_path = [
        [margin, margin],
        [arena_size - margin, margin],
        [arena_size - margin, arena_size - margin],
        [margin, arena_size - margin],
        [margin, margin],  # cerrar room
    ]

    wall_points = crear_paredes_v2(
        trayectoria=room_path,
        num_puntos_por_segmento=80
    )

    walls = [
        Obstacle_c(
            x_prop=x,
            y_prop=y,
            arena_size=int(arena_size),
            radius=wall_thickness / 2
        )
        for x, y in wall_points
    ]

    center_obstacle = Obstacle_c(
        x_prop=arena_size / 2,
        y_prop=arena_size / 2,
        arena_size=int(arena_size),
        radius=14.0
    )

    return walls + [center_obstacle]


def scene3(arena_size=200, wall_thickness=2):
    # 1. Room/perímetro
    margin = wall_thickness / 2

    perimeter_path = [
        [margin, margin],
        [arena_size - margin, margin],
        [arena_size - margin, arena_size - margin],
        [margin, arena_size - margin],
        [margin, margin],
    ]

    perimeter_points = crear_paredes_v2(
        trayectoria=perimeter_path,
        num_puntos_por_segmento=80
    )

    perimeter_walls = [
        Obstacle_wall(
            x,
            y,
            int(arena_size),
            wall_thickness / 2,
            i,
            len(perimeter_points)
        )
        for i, (x, y) in enumerate(perimeter_points)
    ]

    # 2. Pared diagonal original
    diagonal_path = [
        [arena_size, 50],
        [125, 125],
        [75, 75],
    ]

    diagonal_points = crear_paredes_v2(
        trayectoria=diagonal_path,
        num_puntos_por_segmento=50
    )

    diagonal_walls = [
        Obstacle_wall(
            x,
            y,
            int(arena_size),
            wall_thickness / 2,
            i,
            len(diagonal_points)
        )
        for i, (x, y) in enumerate(diagonal_points)
    ]

    # 3. Obstáculo circular central
    center_obstacle = Obstacle_c(
        x_prop=arena_size / 2,
        y_prop=arena_size / 2,
        arena_size=int(arena_size),
        radius=12.0,
    )

    return perimeter_walls + diagonal_walls + [center_obstacle]