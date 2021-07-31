import yaml


class ApplicationSettings:
    def __init__(self, config):
        self.time_limit = float(config.get("time_limit", 600.0))

    def write(self):
        config = {}
        config["time_limit"] = self.time_limit
        return config

class WorldSettings:
    def __init__(self, config):
        self.size_x = float(config.get("size_x", 50.0))
        self.size_y = float(config.get("size_x", 50.0))

        def_spawn_points = [{"x": 45.0, "y": 5.0},
                            {"x": 45.0, "y": 25.0},
                            {"x": 5.0, "y": 45.0},
                            {"x": 10.0, "y": 15.0}]
        spawn_points = config.get("spawn_points", def_spawn_points)
        out_spawn = []
        for pt in spawn_points:
            out_spawn.append((float(pt["x"]), float(pt["y"])))
        self.spawn_points = out_spawn
        
        def_goal_points = [{"x": 5.0, "y": 5.0}, 
                           {"x": 25.0, "y": 25.0},
                           {"x": 15, "y": 40},
                           {"x": 45, "y": 45},
                           {"x": 40, "y": 15}]
        goal_points = config.get("goal_points", def_goal_points)
        out_goal = []
        for pt in goal_points:
            out_goal.append((float(pt["x"]), float(pt["y"])))
        self.goal_points = out_goal

    def write(self):
        config = {}
        config["size_x"] = self.size_x
        config["size_y"] = self.size_y
        config["spawn_points"] = [{"x": p[0], "y": p[1]} for p in self.spawn_points]
        config["goal_points"] = [{"x": p[0], "y": p[1]} for p in self.goal_points]
        return config

class GraphicsSettings:
    def __init__(self, config):
        self.pixels_per_m = int(config.get("pixels_per_m", 10))
        self.show_ms = int(config.get("show_ms", 100))
        self.draw_lidar = bool(config.get("draw_lidar", False))
        self.draw_goals = bool(config.get("draw_goals", True))
    
    def write(self):
        config = {}
        config["pixels_per_m"] = self.pixels_per_m
        config["show_ms"] = self.show_ms
        config["draw_lidar"] = self.draw_lidar
        config["draw_goals"] = self.draw_goals
        return config

class InitialCarSettings:
    def __init__(self, config):
        self.keyboard_cars = min(1, int(config.get("keyboard_cars",0)))
        self.network_cars = int(config.get("network_cars", 1))
        self.random_cars = int(config.get("random_cars", 0))
        self.feedback_cars = int(config.get("feedback_cars", 1))
    
    def write(self):
        config = {}
        config["keyboard_cars"] = self.keyboard_cars
        config["network_cars"] = self.network_cars
        config["random_cars"] = self.random_cars
        config["feedback_cars"] = self.feedback_cars
        return config

class CarProperties:
    def __init__(self, config):
        self.length = float(config.get("length", 4.0))
        self.width = float(config.get("width", 1.8))
        self.mass = float(config.get("mass", 1.0))
        self.fric = float(config.get("fric", 1.0))
        def_lidar_angles = [-0.5, 0.0, 0.5]
        lidar_angles = config.get("lidar_angles", def_lidar_angles)
        self.lidar_angles = [float(a) for a in lidar_angles]

    def write(self):
        config = {}
        config["length"] = self.length
        config["width"] = self.width
        config["mass"] = self.mass
        config["fric"] = self.fric
        config["lidar_angles"] = self.lidar_angles
        return config

class Keyboard:
    def __init__(self, config):
        self.angle = float(config.get("angle", 0.5))
        self.force = float(config.get("force", 5.0))
    
    def write(self):
        config = {}
        config["angle"] = self.angle
        config["force"] = self.force
        return config

class FeedbackCar:
    def __init__(self, config):
        self.left_lidars = [int(x) for x in config.get("left_lidars", [0])]
        self.front_lidars = [int(x) for x in config.get("front_lidars", [1])]
        self.right_lidars = [int(x) for x in config.get("right_lidars", [2])]
        self.force = float(config.get("force", 3.0))
        self.close = float(config.get("close", 5.0))
        self.k = float(config.get("k", 0.25))
    
    def write(self):
        config = {}
        config["left_lidars"] = self.left_lidars
        config["front_lidars"] = self.front_lidars
        config["right_lidars"] = self.right_lidars
        config["force"] = self.force
        config["close"] = self.close
        config["k"] = self.k
        return config

class Physics:
    def __init__(self, config):
        self.physics_timestep = float(config.get("phys_timestep", 0.05))
        self.control_timestep = float(config.get("ctrl_timestep", 0.1))
    
    def write(self):
        config = {}
        config["phys_timestep"] = self.physics_timestep
        config["ctrl_timestep"] = self.control_timestep
        return config

class Preprocessing:
    def __init__(self, config):
        self.gamma = float(config.get("gamma", 0.9))
    
    def write(self):
        config = {}
        config["gamma"] = self.gamma
        return config

class Learning:
    def __init__(self, config):
        self.gamma = float(config.get("gamma", 0.9999))
        self.alpha = float(config.get("alpha", 1e-3))
        self.max_episode_length = float(config.get("max_ep_length", 100))

    def write(self):
        config = {}
        config["gamma"] = self.gamma
        config["alpha"] = self.alpha
        config["max_ep_length"] = self.max_episode_length
        return config

class Statistics:
    def __init__(self, config):
        self.sigma = float(config.get("sigma", 0.1))

    def write(self):
        config = {}
        config["sigma"] = self.sigma
        return config

class Walls:
    def __init__(self, world, config):
        wall_upper = [[0.0,0.0],[world.size_x, 0.0]]
        wall_left = [[0.0,0.0],[0.0, world.size_y]]
        wall_lower = [[0.0, world.size_y],[world.size_x,world.size_y]]
        wall_right= [[world.size_x, 0.0],[world.size_x,world.size_y]]
        self.outer_walls = [wall_upper, wall_lower, wall_left, wall_right]

        # x1y1x2y2
        def_walls = [{"x1": 20, "y1": 10,"x2": 20, "y2": 15},
                     {"x1": 10, "y1": 25,"x2": 15, "y2": 25},
                     {"x1": 30, "y1": 40,"x2": 40, "y2": 30}]
        walls = config.get("walls", def_walls)
        proc_walls = []
        print(walls)
        for w in walls:
            print(w)
            x1 = float(w["x1"])
            y1 = float(w["y1"])
            x2 = float(w["x2"])
            y2 = float(w["y2"])
            proc_walls.append([[x1, y1], [x2, y2]])
        self.walls = proc_walls + self.outer_walls

    def write(self):
        config = {}
        walls = []
        for w in self.walls:
            if w in self.outer_walls:
                continue

            wall = {}
            wall["x1"] = w[0][0]
            wall["y1"] = w[0][1]
            wall["x2"] = w[1][0]
            wall["y2"] = w[1][1]
            walls.append(wall)

        config["walls"] = walls
        return config

class Memory:
    def __init__(self, config):
        self.min_reduce_size = int(config.get("min_reduce_size", 50))
        self.min_train_size = int(config.get("min_train_size", 25))
        self.load_saved_experience = bool(config.get("load_saved_exp", True))
        self.load_saved_network = bool(config.get("load_saved_net", True))
        self.merge_saved_experience = bool(config.get("merged_saved_exp", True))
        self.purge_merged_experience = bool(config.get("purge_merged_exp", False))

    def write(self):
        config = {}
        config["min_reduce_size"] = self.min_reduce_size
        config["min_train_size"] = self.min_train_size
        config["load_saved_exp"] = self.load_saved_experience
        config["load_saved_net"] = self.load_saved_network
        config["merged_saved_exp"] = self.merge_saved_experience
        config["purge_merged_exp"] = self.purge_merged_experience
        return config

class Files:
    def __init__(self, root_dir):
        self.root_dir = root_dir

class Settings:
    def __init__(self, root_dir, settings_file, save_default):
        self.settings_file = settings_file

        config = {}
        try:
            with open(self.settings_file, 'r') as handle:
                config = yaml.load(handle)
        except Exception as e:
            print(e)
            config = {}

        self._files = Files(root_dir)

        self.world = WorldSettings(config.get("world", {}))
        self.graphics = GraphicsSettings(config.get("graphics", {}))
        self.application = ApplicationSettings(config.get("application", {}))
        
        self.initial_car_settings = InitialCarSettings(config.get("init_car", {}))
        self.car_properties = CarProperties(config.get("car_properties", {}))
        self.keyboard = Keyboard(config.get("keyboard", {}))
        self.feedback_car = FeedbackCar(config.get("feedback_car", {}))
        self.physics = Physics(config.get("physics", {}))
        self.walls = Walls(self.world, config.get("walls", {}))
        self.preprocessing = Preprocessing(config.get("preprocessing", {}))
        self.learning = Learning(config.get("learning", {}))
        self.statistics = Statistics(config.get("statistics", {}))
        self.memory = Memory(config.get("memory", {}))

        if save_default:
            self.write()

    def write(self):
        config = {}
        config["world"] = self.world.write()
        config["graphics"] = self.graphics.write()
        config["application"] = self.application.write()
        config["init_car"] = self.car_properties.write()
        config["keyboard"] = self.keyboard.write()
        config["feedback_car"] = self.feedback_car.write()
        config["physics"] = self.physics.write()
        config["walls"] = self.walls.write()
        config["preprocessing"] = self.preprocessing.write()
        config["learning"] = self.learning.write()
        config["statistics"] = self.statistics.write()
        config["memory"] = self.memory.write()

        with open(self.settings_file, 'w') as handle:
            yaml.dump(config, handle)
