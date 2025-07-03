# 每艘飞船的载人和载物容量
people_per_ship = 100
supplies_per_ship = 50

# 总移民人数和关键物资数量
total_people = 1000
total_supplies = 1000 / 100 * 50  # 假设每个人需要0.5单位物资

# 计算所需飞船数量
ships_for_people = (total_people + people_per_ship - 1) // people_per_ship
ships_for_supplies = (total_supplies + supplies_per_ship - 1) // supplies_per_ship

# 取两者的最大值
total_ships_needed = max(ships_for_people, ships_for_supplies)

print(total_ships_needed)
