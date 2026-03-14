import random
import numpy as np
from math import sqrt
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据配置 ====================
islands_data = {
    '岛1': {'lon': 12, 'lat': 0, 'distance': 12,
            'severe': 8, 'mild': 15, 'safe': 42,
            'U': 2, 'T': 5, 'L': 8, 'delta': 0.4, 'theta': 1.65},
    '岛2': {'lon': 0, 'lat': 25, 'distance': 25,
            'severe': 5, 'mild': 12, 'safe': 33,
            'U': 1.5, 'T': 4, 'L': 7, 'delta': 0.5, 'theta': 1.59},
    '岛3': {'lon': 15, 'lat': 18, 'distance': 23.4,
            'severe': 9, 'mild': 20, 'safe': 58,
            'U': 3, 'T': 6, 'L': 10, 'delta': 0.4, 'theta': 2.07},
    '岛4': {'lon': -5, 'lat': 30, 'distance': 30.4,
            'severe': 3, 'mild': 7, 'safe': 15,
            'U': 1, 'T': 3, 'L': 5, 'delta': 0.5, 'theta': 2.28},
    '岛5': {'lon': 20, 'lat': 15, 'distance': 25,
            'severe': 6, 'mild': 10, 'safe': 34,
            'U': 2.5, 'T': 5.5, 'L': 9, 'delta': 0.4, 'theta': 1.00},
    '岛6': {'lon': -8, 'lat': 20, 'distance': 21.5,
            'severe': 4, 'mild': 8, 'safe': 28,
            'U': 1.8, 'T': 4.5, 'L': 7.5, 'delta': 0.5, 'theta': 1.43},
    '岛7': {'lon': 10, 'lat': 30, 'distance': 31.6,
            'severe': 10, 'mild': 18, 'safe': 52,
            'U': 2.2, 'T': 5.2, 'L': 8.5, 'delta': 0.4, 'theta': 1.95},
    '岛8': {'lon': -15, 'lat': 10, 'distance': 18,
            'severe': 2, 'mild': 5, 'safe': 18,
            'U': 1.2, 'T': 3.5, 'L': 6, 'delta': 0.5, 'theta': 1.44},
    '岛9': {'lon': 5, 'lat': 35, 'distance': 35.4,
            'severe': 7, 'mild': 13, 'safe': 40,
            'U': 2.8, 'T': 6.2, 'L': 9.5, 'delta': 0.4, 'theta': 1.54},
    '岛10': {'lon': 25, 'lat': 5, 'distance': 25.5,
             'severe': 9, 'mild': 16, 'safe': 45,
             'U': 2.3, 'T': 5.3, 'L': 8.3, 'delta': 0.4, 'theta': 1.86},
    '安全集散中心': {'lon': 0, 'lat': 0, 'distance': 0}
}

vehicles = {
    'ship': {
        'capacity': 200,
        'speed': 30,
        'cost_per_h': 400,
        'fixed_cost': 2000,
        'time_cost': [300, 3150, 6000],
        'max_count': 10
    },
    'helicopter': {
        'capacity': 20,
        'speed': 150,
        'cost_per_h': 1200,
        'fixed_cost': 5000,
        'time_cost': [800, 7900, 15000],
        'max_count': 10
    }
}

# ==================== DEAP框架配置 ====================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list,
               fitness=creator.FitnessMin,
               ship_routes=list,
               heli_routes=list)

toolbox = base.Toolbox()


# ==================== 遗传操作实现 ====================
def calculate_distance(island1, island2):
    if island1 == '安全集散中心':
        x1, y1 = 0, 0
    else:
        x1, y1 = islands_data[island1]['lon'], islands_data[island1]['lat']

    if island2 == '安全集散中心':
        x2, y2 = 0, 0
    else:
        x2, y2 = islands_data[island2]['lon'], islands_data[island2]['lat']
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def initialize_route(island_keys, vehicle_type):
    """路径初始化"""
    capacity = vehicles[vehicle_type]['capacity']
    max_routes = vehicles[vehicle_type]['max_count']
    routes = []
    remaining = []

    # 处理必须拆分的岛屿
    for island in island_keys:
        load = islands_data[island]['mild'] + islands_data[island]['safe'] if vehicle_type == 'ship' else \
        islands_data[island]['severe']
        if load > capacity:
            required = (load // capacity) + 1
            for _ in range(required):
                if len(routes) < max_routes:
                    routes.append([island])
        else:
            remaining.append(island)

    # 填充剩余岛屿
    for island in remaining:
        placed = False
        for route in routes:
            current_load = sum(
                islands_data[i]['mild'] + islands_data[i]['safe'] if vehicle_type == 'ship' else islands_data[i][
                    'severe'] for i in route)
            if current_load + (
            islands_data[island]['mild'] + islands_data[island]['safe'] if vehicle_type == 'ship' else
            islands_data[island]['severe']) <= capacity:
                route.append(island)
                placed = True
                break
        if not placed and len(routes) < max_routes:
            routes.append([island])

    return routes


def create_individual():
    """个体创建函数"""
    ind = creator.Individual([])

    # 生成有效船只路径
    ship_islands = [k for k in islands_data if
                    k != '安全集散中心' and (islands_data[k]['mild'] + islands_data[k]['safe']) > 0]
    ind.ship_routes = initialize_route(ship_islands, 'ship')

    # 生成有效直升机路径
    heli_islands = [k for k in islands_data if k != '安全集散中心' and islands_data[k]['severe'] > 0]
    ind.heli_routes = initialize_route(heli_islands, 'helicopter')

    return ind


def evaluate(ind):
    """适应度评估函数"""
    # 有效性检查
    if not validate(ind):
        return (float('inf'),)

    # 计算各项成本
    fixed_cost = len(ind.ship_routes) * vehicles['ship']['fixed_cost'] + len(ind.heli_routes) * vehicles['helicopter'][
        'fixed_cost']

    # 时间成本
    ship_time = sum(calculate_route_time(route, 'ship') for route in ind.ship_routes)
    heli_time = sum(calculate_route_time(route, 'helicopter') for route in ind.heli_routes)
    time_cost = ship_time * vehicles['ship']['cost_per_h'] + heli_time * vehicles['helicopter']['cost_per_h']

    # 惩罚成本
    penalty = calculate_penalty(ind.ship_routes, 'ship') + calculate_penalty(ind.heli_routes, 'helicopter')

    total_cost = fixed_cost + time_cost + penalty
    return (total_cost,)


def calculate_route_time(route, vehicle_type):
    """计算路径时间"""
    if not route:
        return 0
    speed = vehicles[vehicle_type]['speed']
    time = 0
    prev = '安全集散中心'
    for island in route:
        time += calculate_distance(prev, island) / speed
        prev = island
    # 返回时间
    time += calculate_distance(route[-1], '安全集散中心') / speed
    return time


def calculate_penalty(routes, vehicle_type):
    """计算时间窗惩罚"""
    total = 0
    speed = vehicles[vehicle_type]['speed']
    for route in routes:
        current_time = 0
        prev = '安全集散中心'
        for island in route:
            dist = calculate_distance(prev, island)
            current_time += dist / speed
            params = islands_data[island]
            sat = time_satisfaction(current_time, params['U'], params['T'], params['L'], params['delta'])

            if sat >= 1:
                cost = 0
            elif sat >= params['delta']:
                cost = vehicles[vehicle_type]['time_cost'][0]
            elif sat > 0:
                cost = vehicles[vehicle_type]['time_cost'][1]
            else:
                cost = vehicles[vehicle_type]['time_cost'][2]

            total += cost * params['theta']
            prev = island
    return total


def time_satisfaction(t, U, T, L, delta):
    """时间满意度计算"""
    if t <= U:
        return 1
    elif U < t <= T:
        return (delta - 1) * ((t - U) / (T - U)) ** 0.5 + 1
    elif T < t <= L:
        return delta * ((t - L) / (T - L)) ** 2
    else:
        return 0


def validate(ind):
    """解的有效性验证"""
    # 检查路径数量限制
    if len(ind.ship_routes) > vehicles['ship']['max_count'] or len(ind.heli_routes) > vehicles['helicopter'][
        'max_count']:
        return False

    # 检查容量约束
    ship_cap = vehicles['ship']['capacity']
    for route in ind.ship_routes:
        if sum(islands_data[island]['mild'] + islands_data[island]['safe'] for island in route) > ship_cap:
            return False

    heli_cap = vehicles['helicopter']['capacity']
    for route in ind.heli_routes:
        if sum(islands_data[island]['severe'] for island in route) > heli_cap:
            return False

    # 检查覆盖所有需求
    total_ship = sum(sum(islands_data[island]['mild'] + islands_data[island]['safe'] for island in route) for route in
                     ind.ship_routes)
    required_ship = sum(islands_data[island]['mild'] + islands_data[island]['safe'] for island in islands_data if
                        island != '安全集散中心')
    total_heli = sum(sum(islands_data[island]['severe'] for island in route) for route in ind.heli_routes)
    required_heli = sum(islands_data[island]['severe'] for island in islands_data if island != '安全集散中心')

    return total_ship >= required_ship and total_heli >= required_heli


def cx_safe_crossover(ind1, ind2):
    """交叉操作"""
    # 船只路径交叉
    if len(ind1.ship_routes) > 1 and len(ind2.ship_routes) > 1:
        cx_point = random.randint(1, min(len(ind1.ship_routes), len(ind2.ship_routes)) - 1)
        ind1.ship_routes[cx_point:], ind2.ship_routes[cx_point:] = ind2.ship_routes[cx_point:], ind1.ship_routes[
                                                                                                cx_point:]

    # 直升机路径交叉
    if len(ind1.heli_routes) > 1 and len(ind2.heli_routes) > 1:
        cx_point = random.randint(1, min(len(ind1.heli_routes), len(ind2.heli_routes)) - 1)
        ind1.heli_routes[cx_point:], ind2.heli_routes[cx_point:] = ind2.heli_routes[cx_point:], ind1.heli_routes[
                                                                                                cx_point:]

    return ind1, ind2


def mut_adaptive_mutation(ind):
    """变异操作"""

    def mutate_route(route):
        if len(route) > 1:
            mutation_type = random.choice(['swap', 'reverse', 'relocate'])
            if mutation_type == 'swap':
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            elif mutation_type == 'reverse':
                start, end = sorted(random.sample(range(len(route)), 2))
                route[start:end + 1] = reversed(route[start:end + 1])
            elif mutation_type == 'relocate':
                idx = random.randint(0, len(route) - 1)
                elem = route.pop(idx)
                route.insert(random.randint(0, len(route)), elem)
        return route

    # 变异船只路径
    for i in range(len(ind.ship_routes)):
        if random.random() < 0.2:
            ind.ship_routes[i] = mutate_route(ind.ship_routes[i].copy())

    # 变异直升机路径
    for i in range(len(ind.heli_routes)):
        if random.random() < 0.2:
            ind.heli_routes[i] = mutate_route(ind.heli_routes[i].copy())

    return ind,


# ==================== 注册遗传操作 ====================
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", cx_safe_crossover)
toolbox.register("mutate", mut_adaptive_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)


# ==================== 可视化函数 ====================
def plot_routes(solution):
    """绘制路径图"""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制所有岛屿
    for name, data in islands_data.items():
        if name == '安全集散中心':
            plt.plot(data['lon'], data['lat'], 'ro', markersize=12, label='安全集散中心')
        else:
            plt.plot(data['lon'], data['lat'], 'o', color='orange', markersize=8)
            plt.text(data['lon'], data['lat'] + 0.5, name, fontsize=10, ha='center', va='bottom')

    # 颜色映射
    ship_colors = plt.cm.tab10(np.linspace(0, 1, len(solution.ship_routes)))
    heli_colors = plt.cm.Set2(np.linspace(0, 1, len(solution.heli_routes)))

    # 绘制船只路径
    for i, route in enumerate(solution.ship_routes):
        path = ['安全集散中心'] + route + ['安全集散中心']
        for j in range(len(path) - 1):
            start = islands_data[path[j]]
            end = islands_data[path[j + 1]]
            plt.plot([start['lon'], end['lon']], [start['lat'], end['lat']],
                     color=ship_colors[i], linestyle='-', linewidth=2,
                     label=f'船只{i + 1}' if j == 0 else "")

    # 绘制直升机路径
    for i, route in enumerate(solution.heli_routes):
        path = ['安全集散中心'] + route + ['安全集散中心']
        for j in range(len(path) - 1):
            start = islands_data[path[j]]
            end = islands_data[path[j + 1]]
            plt.plot([start['lon'], end['lon']], [start['lat'], end['lat']],
                     color=heli_colors[i], linestyle='--', linewidth=2,
                     label=f'直升机{i + 1}' if j == 0 else "")

    plt.title('应急撤离路径规划图', fontsize=14)
    plt.xlabel('经度 (km)', fontsize=12)
    plt.ylabel('纬度 (km)', fontsize=12)
    plt.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    plt.legend(*zip(*unique), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_ship_routes(solution):
    """单独绘制船只路径"""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制所有岛屿
    for name, data in islands_data.items():
        if name == '安全集散中心':
            plt.plot(data['lon'], data['lat'], 'ro', markersize=12, label='安全集散中心')
        else:
            plt.plot(data['lon'], data['lat'], 'o', color='orange', markersize=8)
            plt.text(data['lon'], data['lat'] + 0.5, name,
                     fontsize=10, ha='center', va='bottom')

    # 船只路径颜色
    ship_colors = plt.cm.tab10(np.linspace(0, 1, len(solution.ship_routes)))

    # 绘制船只路径
    for i, route in enumerate(solution.ship_routes):
        path = ['安全集散中心'] + route + ['安全集散中心']
        for j in range(len(path) - 1):
            start = islands_data[path[j]]
            end = islands_data[path[j + 1]]
            plt.plot([start['lon'], end['lon']], [start['lat'], end['lat']],
                     color=ship_colors[i], linestyle='-', linewidth=2,
                     label=f'船只{i + 1}' if j == 0 else "")

    plt.title('船只撤离路径规划', fontsize=14)
    plt.xlabel('经度 (km)', fontsize=12)
    plt.ylabel('纬度 (km)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 处理图例
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.add(l)]
    plt.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def plot_heli_routes(solution):
    """单独绘制直升机路径"""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制所有岛屿
    for name, data in islands_data.items():
        if name == '安全集散中心':
            plt.plot(data['lon'], data['lat'], 'ro', markersize=12, label='安全集散中心')
        else:
            plt.plot(data['lon'], data['lat'], 'o', color='orange', markersize=8)
            plt.text(data['lon'], data['lat'] + 0.5, name,
                     fontsize=10, ha='center', va='bottom')

    # 直升机路径颜色
    heli_colors = plt.cm.Set2(np.linspace(0, 1, len(solution.heli_routes)))

    # 绘制直升机路径
    for i, route in enumerate(solution.heli_routes):
        path = ['安全集散中心'] + route + ['安全集散中心']
        for j in range(len(path) - 1):
            start = islands_data[path[j]]
            end = islands_data[path[j + 1]]
            plt.plot([start['lon'], end['lon']], [start['lat'], end['lat']],
                     color=heli_colors[i], linestyle='--', linewidth=2,
                     label=f'直升机{i + 1}' if j == 0 else "")

    plt.title('直升机撤离路径规划', fontsize=14)
    plt.xlabel('经度 (km)', fontsize=12)
    plt.ylabel('纬度 (km)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 处理图例
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.add(l)]
    plt.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def print_detailed_solution(solution):
    """打印解决方案详情"""
    print("\n" + "=" * 60)
    print(" " * 20 + "最优撤离方案详情")
    print("=" * 60)

    # 船只运输详情
    print("\n【船只运输方案】")
    ship_total = 0
    for i, route in enumerate(solution.ship_routes, 1):
        load = sum(islands_data[island]['mild'] + islands_data[island]['safe'] for island in route)
        time = calculate_route_time(route, 'ship')
        print(f"\n船只{i}路径: 安全集散中心 → {' → '.join(route)} → 安全集散中心")
        print(f"  载客量: {load}人 | 预计耗时: {time:.2f}小时")
        ship_total += load

    # 直升机运输详情
    print("\n【直升机运输方案】")
    heli_total = 0
    for i, route in enumerate(solution.heli_routes, 1):
        load = sum(islands_data[island]['severe'] for island in route)
        time = calculate_route_time(route, 'helicopter')
        print(f"\n直升机{i}路径: 安全集散中心 → {' → '.join(route)} → 安全集散中心")
        print(f"  载客量: {load}人 | 预计耗时: {time:.2f}小时")
        heli_total += load

    # 成本分析
    print("\n" + "=" * 60)
    print("【成本分析】")
    fixed_cost = len(solution.ship_routes) * vehicles['ship']['fixed_cost'] + len(solution.heli_routes) * \
                 vehicles['helicopter']['fixed_cost']
    time_cost = sum(calculate_route_time(r, 'ship') * vehicles['ship']['cost_per_h'] for r in solution.ship_routes) + \
                sum(calculate_route_time(r, 'helicopter') * vehicles['helicopter']['cost_per_h'] for r in
                    solution.heli_routes)
    penalty = calculate_penalty(solution.ship_routes, 'ship') + calculate_penalty(solution.heli_routes, 'helicopter')

    print(f"总成本: {solution.fitness.values[0]:.2f}元")
    print(f"  固定成本: {fixed_cost}元")
    print(f"  时间成本: {time_cost:.2f}元")
    print(f"  惩罚成本: {penalty:.2f}元")

    print("\n【验证信息】")
    if validate(solution):
        print("✅ 方案验证通过")
        print(f"总计撤离: {ship_total + heli_total}人 (船只: {ship_total}, 直升机: {heli_total})")
    else:
        print("❌ 方案存在违规")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 运行遗传算法
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.3, ngen=500,
        stats=stats, halloffame=hof, verbose=True
    )

    # 输出结果
    best_ind = population[0]
    print_detailed_solution(best_ind)
    plot_routes(best_ind)
    plot_heli_routes(best_ind)
    plot_ship_routes(best_ind)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 5))
    plt.plot(log.select('gen'), log.select('min'), 'b-')
    plt.title("适应度收敛曲线")
    plt.xlabel("代数")
    plt.ylabel("最小成本")
    plt.grid(True)
    plt.show()