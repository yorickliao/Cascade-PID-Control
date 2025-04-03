# Final UAV controller using Cascade PID + yaw compensation
# Based on the structure of your 2D twin-motor control logic

import math

# === Global states ===
prev_error_pos = [0.0, 0.0, 0.0]  # 上一次位置误差 [x, y, z]
integral_error_pos = [0.0, 0.0, 0.0]  # 积分误差（用于Ki）

prev_error_yaw = 0.0
integral_error_yaw = 0.0

last_target = [None, None, None, None]

def controller(state, target_pos, dt):
    """
    UAV controller: Cascade PID (position control) + yaw control + yaw compensation (world to body frame)
    Input:
        state = [x, y, z, roll, pitch, yaw]
        target_pos = [target_x, target_y, target_z, target_yaw]
        dt = timestep (s)
    Output:
        velocity command: (vx, vy, vz, yaw_rate)
    """
    global prev_error_pos, integral_error_pos
    global prev_error_yaw, integral_error_yaw
    global last_target

    # === Extract state 当前无人机状态 ===
    x, y, z, roll, pitch, yaw = state
    target_x, target_y, target_z, target_yaw = target_pos

    # === 如果目标改变，重置误差项 ===
    if target_pos != last_target:
        prev_error_pos = [0.0, 0.0, 0.0]
        integral_error_pos = [0.0, 0.0, 0.0]
        prev_error_yaw = 0.0
        integral_error_yaw = 0.0
        last_target = list(target_pos)

    # === PID 参数（已调优，适合平稳稳定控制） ===
    Kp = 0.25
    Ki = 0.005
    Kd = 0.01

    Kp_yaw = 2.0
    Ki_yaw = 0.02
    Kd_yaw = 0.1

    # === 位置误差计算 ===
    error_x = target_x - x
    error_y = target_y - y
    error_z = target_z - z

    # === 积分误差累加 ===
    integral_error_pos[0] += error_x * dt
    integral_error_pos[1] += error_y * dt
    integral_error_pos[2] += error_z * dt

    # === 限制积分项（防止风up）===
    max_integral = 0.5
    for i in range(3):
        integral_error_pos[i] = max(min(integral_error_pos[i], max_integral), -max_integral)

    # === 微分误差 ===
    derivative_x = (error_x - prev_error_pos[0]) / dt
    derivative_y = (error_y - prev_error_pos[1]) / dt
    derivative_z = (error_z - prev_error_pos[2]) / dt

    # === PID 控制输出（世界坐标系） ===
    vx = Kp * error_x + Ki * integral_error_pos[0] + Kd * derivative_x
    vy = Kp * error_y + Ki * integral_error_pos[1] + Kd * derivative_y
    vz = Kp * error_z + Ki * integral_error_pos[2] + Kd * derivative_z

    prev_error_pos = [error_x, error_y, error_z]

    # === 限制速度输出（更平稳） ===
    vx = max(min(vx, 0.5), -0.5)
    vy = max(min(vy, 0.5), -0.5)
    vz = max(min(vz, 0.5), -0.5)

    # === 航向角 yaw 控制 ===
    yaw_error = target_yaw - yaw
    while yaw_error > math.pi:
        yaw_error -= 2 * math.pi
    while yaw_error < -math.pi:
        yaw_error += 2 * math.pi

    integral_error_yaw += yaw_error * dt
    derivative_yaw = (yaw_error - prev_error_yaw) / dt

    yaw_rate = (
        Kp_yaw * yaw_error
        + Ki_yaw * integral_error_yaw
        + Kd_yaw * derivative_yaw
    )

    prev_error_yaw = yaw_error

    # === 限制航向角速度 ===
    yaw_rate = max(min(yaw_rate, 1.74533), -1.74533)  # ±100 deg/s

    # === 坐标转换（从世界系到机体系） ===
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    vx_body = cos_yaw * vx + sin_yaw * vy
    vy_body = -sin_yaw * vx + cos_yaw * vy

    # === 返回控制命令（机体系）===
    return (vx_body, vy_body, vz, yaw_rate)
