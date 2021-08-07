import math
import numpy as np

from .car import Car, CarState

class PhysicsEngine:
    def __init__(self, settings, world, experience):
        self.settings = settings
        self.world = world
        self.experience = experience

        self.controllers = None

        self.T_passed = 0.0
    
    def set_controllers(self, controllers):
        self.controllers = controllers

        for car in self.world.keyboard_cars:
            car.set_controller(self.controllers.keyboard)

        for car in self.world.network_cars:
            car.set_controller(self.controllers.network)
        
        for car in self.world.hardcoded_cars:
            car.set_controller(self.controllers.hardcoded)
        
        for car, controller in zip(self.world.random_cars, self.controllers.random):
            car.set_controller(controller)

        for car in self.world.feedback_cars:
            car.set_controller(self.controllers.feedback)

        for car, controller in zip(self.world.network_exploration_cars, self.controllers.network_exploration):
            car.set_controller(controller)

        for car, controller in zip(self.world.feedback_exploration_cars, self.controllers.feedback_exploration):
            car.set_controller(controller)

    def full_control_sensor_step(self):
        self.sensors_step()
        self.experience.sample_end_states()
        self.experience.sample_rewards()

        self.experience.new_experience_step()

        self.experience.handle_episode_ends()
        self.handle_resets()
        
        self.controls_step()

        self.experience.sample_start_states()
        self.experience.sample_controls()

    def full_physics_termination_step(self):
        self.physics_time_step()
        self.handle_goals()
        self.handle_collisions()
    
    def physics_time_step(self):
        for car in self.world.all_cars:
            self.step_car_physics(car)

    def step_car_physics(self, car):
        #print(car.controls.force)
        dt = self.settings.physics.physics_timestep
        v = car.state.v
        #ang = math.atan2(car.state.vy, car.state.vx)
        #ang = ang - car.state.h
        #v = v * math.cos(ang)

        v = v + (car.controls.force - v*self.settings.car_properties.fric) *  dt / self.settings.car_properties.mass

        car.state.dh = float(v * car.controls.steer)
        car.state.h = float(car.state.h + car.state.dh * dt) 

        car.state.v = float(v)
        car.state.x = float(car.state.x + car.state.v * math.cos(car.state.h) * dt)
        car.state.y = float(car.state.y + car.state.v * math.sin(car.state.h) * dt)

    def sensors_step(self):
        for car in self.world.all_cars:
            state = self.get_car_state(car)
            car.sensed_state = state

    def controls_step(self):
        for car in self.world.all_cars:
            state = car.sensed_state
            control = car.get_controls(state)
            car.set_controls(control)

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
        base_state = np.array([car.state.x, car.state.y,car.state.h,dist,head])

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
            did_reset = False

            if car.reached_goal:
                self.new_goal(car)
                car.reached_goal = False
                did_reset = True
            elif car.collided:
                self.respawn_car(car)
                car.collided = False
                did_reset = True

            elif car.too_old:
                self.respawn_car(car)
                car.too_old = False
                did_reset = True

            if did_reset and car in self.world.random_cars:
                i = self.world.random_cars.index(car)
                self.controllers.random[i].reset()

