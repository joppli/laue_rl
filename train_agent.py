#!/usr/bin/env python3
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import ast
import argparse
import warnings

from datetime import datetime
from laue_utils import get_timestamp
from agent_trainer import AgentTrainerOffPolicy   

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("algo", type=str, help="RL algorithm ('drm' or 'sac').")
    parser.add_argument("save_dir", type=str, help="(Base) save directory.")
    parser.add_argument("config_file", type=str, help="Configuration file (.json) containing the run parameters.")

    # Parse known and unknown arguments
    args, uargs = parser.parse_known_args()

    # Parse unknown arguments manually into a dictionary (for --key=value format) 
    # (used to partially overwrite run configuration)
    kwargs = {}
    for i, arg in enumerate(uargs):
        if arg.startswith('--'):
            if '=' in arg:
                key_value = arg.split('=', 1)
            else:
                # check if there is a following value that does not start with a dash
                if i+1 < len(uargs) and not uargs[i+1].startswith('-'):
                    key_value = [arg, uargs[i+1]]
            if len(key_value) == 2:
                key, value = key_value
                try:
                    # Try to safely evaluate Python literals (list, int, float, etc)
                    value = ast.literal_eval(value)
                except:
                    pass
                kwargs[key.lstrip('--')] = value
            else:
                print(f"Skipping invalid argument '{arg}'.")

    # Create new save directory with current timestamp
    save_dir = os.path.join(args.save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Instantiate trainer
    trainer = AgentTrainerOffPolicy(
        algo=args.algo,
        save_dir=save_dir,
        config_file=args.config_file,
        **kwargs
    )

    # Train agent
    trainer.train()

if __name__ == "__main__":
    main()
