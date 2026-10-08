"""Best-effort cleanup, including partially connected LeRobot SO-101 arms."""


def disconnect_arm(arm):
    # SOFollower.is_connected is False if even one camera failed to connect.
    # Disconnect bus and cameras independently so one failure cannot skip others.
    bus = arm.bus
    if bus.is_connected:
        try:
            if hasattr(arm.config, "disable_torque_on_disconnect"):
                bus.disconnect(arm.config.disable_torque_on_disconnect)
            else:
                bus.disconnect()
        except Exception as exc:
            print(f"[WARN] Motor bus disconnect failed: {exc}")
    for name, camera in getattr(arm, "cameras", {}).items():
        try:
            if camera.is_connected:
                camera.disconnect()
        except Exception as exc:
            print(f"[WARN] Camera {name} disconnect failed: {exc}")
