# Allow direct execution from the repository as well as python -m.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time 

from lerobot.teleoperators.so_leader import (
    SO101Leader,
    SO101LeaderConfig
)

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig
)

from hardware_constant import LEADER_PORT, FOLLOWER_PORT, LEADER_ID, FOLLOWER_ID, CONTROL_DT
from hardware_utils import disconnect_arm


def main():

    print("=========================================")
    print("         SO-ARM101 TELEOPERATION         ")
    print("=========================================\n")

    #Config the arm

    leader_config = SO101LeaderConfig(
        port = LEADER_PORT,
        id = LEADER_ID,
    )

    follower_config = SO101FollowerConfig(
        port = FOLLOWER_PORT,
        id = FOLLOWER_ID,
        disable_torque_on_disconnect=True,
    )

    leader = SO101Leader(leader_config)
    follower = SO101Follower(follower_config)

    exit_reason = "user"

    # Teleoperate
    try: 
        print("Connecting follower....")
        follower.connect()

        print("Connecting leader...")
        leader.connect()

        print("Teleoperation started. Press Ctrl+C to stop")

        while True:
            start_time = time.perf_counter()

            # Read the target position from leader 
            try:
                leader_state = leader.get_action()
            except Exception as e:
                print(f"\n[ERROR] Leader arm was disconnected: {e}")
                exit_reason = "LEADER DISCONNECTED"
                break
 
            # Send target
            try:
                follower.send_action(leader_state)

            except Exception as e:
                print(f"\n[ERROR] Failed to command follower: {e}")
                exit_reason = "FOLLOWER WRITE ERROR"
                break


            # Control loop in CONTROL_HZ
            cur_time = time.perf_counter() - start_time

            remaining_time = CONTROL_DT - cur_time

            if remaining_time > 0:
                time.sleep(remaining_time)

            else:
                print(
                    f"Warning slow loop "
                    f"{cur_time * 1000:.1f} ms"
                )
        
    except KeyboardInterrupt:
        exit_reason = "KEYBOARD INTERRUPT"
        print("\nUser Stop Teleoperation...")

    except Exception as e:
        exit_reason = "UNEXPECTED ERROR"
        print(f"\n[UNEXPECTED ERROR] {type(e).__name__}: {e}")

    finally:

        print(f"\nExit reason: {exit_reason}")

        disconnect_arm(follower)
        disconnect_arm(leader)

        print("Disconnected")


if __name__ == "__main__":
    main()
