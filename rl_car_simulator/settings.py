


class ApplicationSettings:
    def __init__(self):
        self.time_limit = 300.0

class WorldSettings:
    def __init__(self):
        self.size_x = 50.0
        self.size_y = 50.0

        self.spawn_points = [(5.0, 10.0),
                             (30.0, 30.0),
                             (25.0, 25.0),
                             (5.0, 20.0, 2.0)]
        self.goal_points = [(3.0,3.0), (20.0, 20.0)]

class GraphicsSettings:
    def __init__(self):
        self.pixels_per_m = 20
        self.show_ms = 100

class InitialCarSettings:
    def __init__(self):
        self.keyboard_cars = 0
        self.network_cars = 1

class CarProperties:
    def __init__(self):
        self.length = 4.0
        self.width = 1.8
        self.mass = 1.0
        self.fric = 1.0
        self.lidar_angles = [0.0, 0.5, -0.5]

class Keyboard:
    def __init__(self):
        self.angle = 0.5
        self.force = 5.0

class Physics:
    def __init__(self):
        self.physics_timestep = 0.05
        self.control_timestep = 0.1

class Preprocessing:
    def __init__(self):
        self.gamma = 0.9

class Learning:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.max_episode_length = 50

class Walls:
    def __init__(self):
        wall_upper = ((0.0,0.0),(WorldSettings().size_x, 0.0))
        wall_left = ((0.0,0.0),(0.0, WorldSettings().size_y))
        wall_lower = ((0.0, WorldSettings().size_y),(WorldSettings().size_x,WorldSettings().size_y))
        wall_right= ((WorldSettings().size_x, 0.0),(WorldSettings().size_x,WorldSettings().size_y))
        self.walls = [((5,5),(20,5))]
        self.walls = self.walls + [wall_upper, wall_lower, wall_left, wall_right]

class Settings:
    def __init__(self):
        self.world = WorldSettings()
        self.graphics = GraphicsSettings()
        self.application = ApplicationSettings()
        
        self.initial_car_settings = InitialCarSettings()
        self.car_properties = CarProperties()
        self.keyboard = Keyboard()
        self.physics = Physics()
        self.walls = Walls()
        self.preprocessing = Preprocessing()
        self.learning = Learning()