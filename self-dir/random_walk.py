import random

# 地图的宽度和高度
width = 80
height = 25

# 地图初始化为全是墙
map = [['#' for y in range(height)] for x in range(width)]

# 随机选择一个起始点
x, y = random.randint(0, width - 1), random.randint(0, height - 1)

# 定义四个可能的移动方向
directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

for i in range(10000):  # 进行10000步随机漫步
    dx, dy = random.choice(directions)  # 随机选择一个方向
    x += dx
    y += dy
    # 确保漫步不会超出地图边界
    if x < 0: x = 0
    if x >= width: x = width - 1
    if y < 0: y = 0
    if y >= height: y = height - 1
    # 在地图上留下路径
    map[x][y] = '.'

# 打印地图
for row in map:
    print(''.join(row))
