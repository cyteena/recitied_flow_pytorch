"""
Profile Training Script for Rectified Flow

This script runs the training with profiling enabled to identify performance bottlenecks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_oxford import Config
import tyro


def profile_main():
    """Main function with profiling enabled."""

    # Parse config
    config = tyro.cli(Config)

    # Enable profiling by setting an environment variable or modifying the config
    # We'll modify the Trainer class to include profiling when this script is run
    os.environ['RECTIFIED_FLOW_PROFILE'] = '1'

    from train_oxford import main
    main(config)


if __name__ == "__main__":
    print("Starting training with profiling enabled...")
    print("This will measure the time spent in different training components.")
    print("Profiling reports will be shown periodically and at the end.\n")

    profile_main()
