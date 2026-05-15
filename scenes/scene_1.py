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