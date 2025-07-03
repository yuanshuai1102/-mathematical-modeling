# 人数和物资参数
total_people = 1000
people_per_ship = 100
supplies_per_ship = 50

# 物资和航天器限制
max_supplies = supplies_per_ship * (total_people // people_per_ship)
max_ships = (total_people // people_per_ship) + 1

# 初始化动态规划表
dp = [[float('inf')] * (max_supplies + 1) for _ in range(max_ships + 1)]
dp[0][0] = 0

# 动态规划计算
for i in range(1, max_ships + 1):
    for j in range(max_supplies + 1):
        if j >= supplies_per_ship:
            dp[i][j] = min(dp[i][j], dp[i-1][j-supplies_per_ship] + people_per_ship)

# 找到满足条件的最少航天器数量
min_ships = float('inf')
for j in range(max_supplies + 1):
    if dp[max_ships][j] >= total_people:
        min_ships = min(min_ships, max_ships)

print(min_ships)
