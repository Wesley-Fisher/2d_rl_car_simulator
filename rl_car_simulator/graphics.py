import numpy as np
import cv2
import copy

class Graphics:
    def __init__(self, settings, world):
        self.settings = settings
        self.world = world

        w = int(self.settings.world.size_x * self.settings.graphics.pixels_per_m)
        h = int(self.settings.world.size_y * self.settings.graphics.pixels_per_m)
        base_world_frame = np.full((h,w,3), (255,255,255), dtype=np.uint8)
        self.base_world_frame = self.fill_base_frame(base_world_frame)

    def close(self):
        cv2.destroyAllWindows()
    
    def fill_base_frame(self, frame):
        for wall in self.world.walls:
            self.draw_wall(frame, wall)
        for goal in self.settings.world.goal_points:
            self.draw_goal(frame, goal)
        return frame

    def draw_wall(self, frame, wall):
        p1 = (wall.x1 * self.settings.graphics.pixels_per_m).astype(np.int32)
        p1 = tuple(p1.tolist())
        p2 = (wall.x2 * self.settings.graphics.pixels_per_m).astype(np.int32)
        p2 = tuple(p2.tolist())
        cv2.line(frame, p1, p2, color=(0,0,0), thickness=4)

    def draw_goal(self, frame, goal):
        goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in goal])
        cv2.circle(frame, goal, color=(0,255,0), radius=25, thickness=-1)

    def create_current_frame(self, base_frame):

        for car in self.world.keyboard_cars:
            self.draw_car(base_frame, car, (255,0,0))

            goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in car.goal])
            cv2.circle(base_frame, goal, color=(0,215,255), radius=10, thickness=-1)

        for car in self.world.network_cars:
            self.draw_car(base_frame, car, (0,0,255))
            
            curr = tuple([int(c* self.settings.graphics.pixels_per_m) for c in [car.state.x, car.state.y]])
            goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in car.goal])
            cv2.line(base_frame, curr, goal, color=(0,0,0), thickness=1)

        return base_frame

    def draw_car(self, frame, car, color):
        corners = car.get_corners()
        corners = np.concatenate(corners, axis=1).T
        corners = (corners * self.settings.graphics.pixels_per_m).astype(np.int32)
        cv2.fillPoly(frame, [corners], color)

    def show_current_world(self):
        frame = self.create_current_frame(copy.copy(self.base_world_frame))
        cv2.imshow('test', frame)

    def sleep(self):
        cv2.waitKey(self.settings.graphics.show_ms)
