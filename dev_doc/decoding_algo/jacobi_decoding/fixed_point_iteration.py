# 计算 f(x) = 0的解
# 转换成 g(x) = x - f(x). 不唯一, 比如 x - 0.5f
# 则 f的解, 是g的不动点 fixed point
# f(x0) = 0 ===> g(x0) = x0
import time


def get_fixed_point_equation(x):
    # 要精心控制 fx 防止发散
    # 比如 fx = x*x - 5
    return (x + 5 / x) / 2


def resolve_fx_by_fixed_point_iteration():
    eps = 1e-5
    count = 0
    x = 0.1
    while True:
        y = get_fixed_point_equation(x)
        delta = abs(y - x)
        print(f"delta={delta:.4f}, count={count}, x={x:.4f}, y={y:.4f}")
        if delta < eps:
            break
        count += 1
        x = y
        # time.sleep(3)


if __name__ == "__main__":
    resolve_fx_by_fixed_point_iteration()
