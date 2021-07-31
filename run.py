#!/usr/bin/env python3
import os
import sys

import rl_car_simulator.game_engine as Engine
from rl_car_simulator.settings import Settings

run_dir = os.path.dirname(os.path.realpath(__file__))
settings_dir = run_dir + "/config"

settings = "default"
if len(sys.argv) == 2:
    settings = sys.argv[1]
if not settings.endswith(".yaml"):
    settings = settings + ".yaml"
settings_file = settings_dir + "/" + settings

settings = Settings(run_dir, settings_file, settings == "default.yaml")
engine = Engine.GameEngine(settings)
engine.run()