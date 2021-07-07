import math
import numpy as np

from .car import Car, CarState

class PhysicsEngine:
    def __init__(self, settings, world):
        self.settings = settings
        self.world = world

        self.controllers = None

        self.T_passed = 0.0
    
    def set_controllers(self, controllers):
        self.controllers = controllers
    
    def physics_time_step(self):
        for car in self.world.all_cars:
            self.step_car_physics(car)

    def step_car_physics(self, car):
        #print(car.controls.force)
        dt = self.settings.physics.physics_timestep
        v = math.sqrt(car.state.vx*car.state.vx + car.state.vy*car.state.vy)
        ang = math.atan2(car.state.vy, car.state.vx)
        ang = ang - car.state.h
        v = v * math.cos(ang)

        v = v + (car.controls.force - v*self.settings.car_properties.fric) *  dt / self.settings.car_properties.mass

        car.state.dh = v * car.controls.steer
        car.state.h = car.state.h + car.state.dh * dt 

        car.state.vx = v * math.cos(car.state.h)
        car.state.vy = v * math.sin(car.state.h)

        car.state.x = car.state.x + car.state.vx * dt
        car.state.y = car.state.y + car.state.vy * dt

    def sensors_step(self):
        for car in self.world.all_cars:
            state = self.get_car_state(car)
            car.sensed_state = state

    def controls_step(self):
        for car in self.world.keyboard_cars:
            car.set_controls(self.controllers.keyboard.get_controls(car.sensed_state))

        for car in self.world.network_cars:
            state = self.get_car_state(car)
            car.set_controls(self.controllers.network.get_controls(car.sensed_state))

        for car in self.world.hardcoded_cars:
            state = self.get_car_state(car)
            car.set_controls(self.controllers.hardcoded.get_controls(car.sensed_state))

    def get_null_car_state(self):
        car = Car()
        base_state = np.array([car.state.x,
                               car.state.y,
                               car.state.h,
                               0.0, # Distance to goal
                               0.0, # Heading to goal])
                                ])
        lidar_state = np.zeros((len(self.settings.car_properties.lidar_angles)))
        return np.concatenate([base_state, lidar_state], axis=0)
    
    def get_car_state(self, car):
        dx = car.goal[0] - car.state.x
        dy = car.goal[1] - car.state.y
        dist = math.sqrt(dx*dx + dy*dy)
        head = math.atan2(dy, dx)
        base_state = np.array([car.state.x,
                               car.state.y,
                               car.state.h,
                               dist, # Distance to goal
                               head, # Heading to goal])
                                ])

        N = len(self.settings.car_properties.lidar_angles)
        lidar_state = np.zeros((N))
        for i in range(0, N):
            ang = self.settings.car_properties.lidar_angles[i]
            dist = self.calc_lidar_distance(car, ang)
            lidar_state[i] = dist
        car.lidar_state = lidar_state
        return np.concatenate([base_state, lidar_state], axis=0)

    def calc_lidar_distance(self, car, ang):
        c = np.array([[car.state.x],[car.state.y]]).reshape(-1)
        beam = np.array([[math.cos(ang + car.state.h)],[math.sin(ang + car.state.h)]])
        
        dist_min = -1
        for wall in self.world.walls:
            v1 = wall.x1 - c
            v2 = wall.x2 - c
            V = np.concatenate([v1.reshape((2,1)),v2.reshape((2,1))], axis=-1)
            coeff = np.dot(np.linalg.pinv(V), beam)
            a = coeff[0][0]
            b = coeff[1][0]
            if a >= 0.0 and a <= 1.0 and  b >= 0.0 and b <= 1.0 and (a+b) > 0.0:
                alpha = 1.0 / (b + a)
                pt = np.dot(V, coeff) * alpha
                diff = pt
                dist = np.linalg.norm(diff, 2)

                if dist_min == -1 or dist < dist_min:
                    dist_min = dist

        return dist_min
    
    def handle_collisions(self):
        for car in self.world.all_cars:
            if self.car_in_collision(car):
                car.collided = True
                
    
    def car_in_collision(self, car):
        for wall in self.world.walls:
            if self.car_wall_collision(car, wall):
                return True
        return False

    def car_wall_collision(self, car, wall):
        points = car.get_corners() + [car.get_center()]

        polarities = set()
        for point in points:
            if wall.point_is_bounded(point):
                p = wall.get_point_side(point)
                polarities.add(p)
            if len(polarities) > 1:
                return True
        return False

    def respawn_car(self, car):
        (x,y,h) = self.world.random_spawn_state()
        cs = CarState()
        cs.x = x
        cs.y = y
        cs.h = h
        car.state = cs

    def handle_goals(self):
        for car in self.world.all_cars:
            if self.car_near_goal(car):
                car.reached_goal = True
                
    
    def car_near_goal(self, car):
        dx = car.state.x - car.goal[0]
        dy = car.state.y - car.goal[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist < 1.0

    def new_goal(self, car):
        goal = None
        id = car.goal_id

        while id == car.goal_id:
            goal, id = self.world.random_goal_state()
        car.set_goal(goal, id)

    def handle_resets(self):
        for car in self.world.all_cars:
            if car.reached_goal:
                self.new_goal(car)
                car.reached_goal = False
            elif car.collided:
                self.respawn_car(car)
                car.collided = False

