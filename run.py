#!/usr/bin/env python3
import os

import rl_car_simulator.game_engine as Engine
from rl_car_simulator.settings import Settings

settings = Settings()
settings.files.root_dir = os.path.dirname(os.path.realpath(__file__))
engine = Engine.GameEngine(settings)
engine.run()