import numpy as np
import matplotlib.pyplot as plt

# Исходные данные — вариант 4
m, D, v0, alpha_deg = 0.98, 0.12, 15.0, 52
Cw, rho, g, dt = 0.25, 1.29, 9.81, 0.001

S = np.pi * (D / 2) ** 2
k = 0.5 * Cw * rho * S

def simulate(alpha, with_drag=True):
    a = np.radians(alpha)
    vx, vy = v0 * np.cos(a), v0 * np.sin(a)
    x = y = t = 0.0
    xs, ys, ts, vxs, vys = [x], [y], [t], [vx], [vy]

    while t < 100:
        speed = np.hypot(vx, vy)
        if with_drag:
            ax = -(k / m) * speed * vx
            ay = -g - (k / m) * speed * vy
        else:
            ax, ay = 0.0, -g

        # Метод Эйлера–Кромера: сначала скорости, затем координаты
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        t_new = t + dt

        if y_new <= 0 and t_new > 0.01:
            # Интерполяция последнего шага до y = 0
            f = (0.0 - y) / (y_new - y)
            xs.append(x + f * (x_new - x))
            ys.append(0.0)
            ts.append(t + f * dt)
            vxs.append(vx + f * (vx_new - vx))
            vys.append(vy + f * (vy_new - vy))
            break

        x, y, vx, vy, t = x_new, y_new, vx_new, vy_new, t_new
        xs.append(x); ys.append(y); ts.append(t)
        vxs.append(vx); vys.append(vy)

    return (np.array(xs), np.array(ys), np.array(ts),
            np.array(vxs), np.array(vys))

# Основной расчёт для заданного угла
xd, yd, td, vxd, vyd = simulate(alpha_deg, True)
xv, yv, tv, vxv, vyv = simulate(alpha_deg, False)

L_drag, H_drag, T_drag = xd[-1], yd.max(), td[-1]
v_end_drag = np.hypot(vxd[-1], vyd[-1])

# Аналитическое решение в вакууме
a = np.radians(alpha_deg)
L_vac = v0**2 * np.sin(2*a) / g
H_vac = (v0*np.sin(a))**2 / (2*g)
T_vac = 2*v0*np.sin(a) / g
v_end_vac = v0

print("Сопротивление воздуха:")
print(f"L = {L_drag:.3f} м, H = {H_drag:.3f} м, "
      f"T = {T_drag:.3f} с, v_end = {v_end_drag:.3f} м/с")
print("Вакуум (аналитически):")
print(f"L = {L_vac:.3f} м, H = {H_vac:.3f} м, "
      f"T = {T_vac:.3f} с, v_end = {v_end_vac:.3f} м/с")

# График сравнения траекторий
plt.figure(figsize=(9, 5))
plt.plot(xd, yd, label="С сопротивлением воздуха")
plt.plot(xv, yv, "--", label="В вакууме")
plt.xlabel("x, м"); plt.ylabel("y, м")
plt.title(f"Траектории полёта, вариант 4 (α = {alpha_deg}°)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("variant4_trajectory.png", dpi=150)
plt.show()

# Поиск оптимального угла от 10° до 80° с шагом 1°
angles = np.arange(10, 81, 1)
ranges = [simulate(float(a), True)[0][-1] for a in angles]
best = int(angles[np.argmax(ranges)])
print(f"Оптимальный угол: {best}°, максимальная дальность: {max(ranges):.3f} м")

plt.figure(figsize=(9, 5))
plt.plot(angles, ranges, label="С сопротивлением воздуха")
plt.axvline(best, linestyle=":", label=f"Оптимальный угол: {best}°")
plt.axvline(45, linestyle="--", label="45° (вакуум)")
plt.xlabel("Угол броска α, °"); plt.ylabel("Дальность L, м")
plt.title("Зависимость дальности от угла броска")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("variant4_range_vs_angle.png", dpi=150)
plt.show()

# Время подъёма и спуска
i_top = int(np.argmax(yd))
t_up = td[i_top]
t_down = T_drag - t_up
print(f"t_up = {t_up:.3f} с; t_down = {t_down:.3f} с")
