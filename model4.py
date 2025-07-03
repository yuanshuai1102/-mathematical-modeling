import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 读取景点数据
city_sights_df = pd.read_csv('top_city_sights.csv')
train_df = pd.read_csv('train (2).csv')

start_station_col = 'start_station_name'
end_station_col = 'end_station_name'
start_time_col = 'start_time'
end_time_col = 'arrive_time'
cost_col = 'zy_price'

# 创建城市到高铁站的映射 (这个需要根据实际数据进行填写)
city_to_station = {
    '广州南': '广州', '北京南': '北京', '重庆北': '重庆', '成都东': '成都', '天津西': '天津', '玉门': '玉门', '吉林': '吉林', '梧州南': '梧州', '南平市': '南平市', '济南西': '济', '巴东': '巴', '南宁': '南宁', '峨眉': '峨眉', '厦门': '厦门', '通化': '通化', '沈阳': '沈阳', '昆明': '昆明', '惠阳': '惠阳', '太湖南': '太湖', '北海': '北海', '宁海': '宁海', '常州北': '常州', '赣州西': '赣州', '烟台': '烟台', '盐城': '盐城', '汉口': '汉口', '咸宁南': '咸宁', '长沙南': '长沙', '湛江西': '湛江', '通辽': '通辽', '徐州': '徐州', '济南': '济', '大连北': '大连', '齐齐哈尔南': '齐齐哈尔', '株洲西': '株洲', '丽水': '丽水', '西安': '西安', '苍南': '苍', '南京': '南京', '杭州西': '杭州', '石柱县': '石柱县', '广州东': '广州', '嘉峪关南': '嘉峪关', '九江': '九江', '宜兴': '宜兴', '泰宁': '泰宁', '银川': '银川', '福州': '福州', '温州': '温州', '荣成': '荣成', '三  亚': '三  亚', '合肥南': '合肥', '曲阜东': '曲阜', '延吉西': '延吉', '盘州': '盘州', '珠海': '珠海', '蚌埠南': '蚌埠', '成都西': '成都', '亳州南': '亳州', '宜昌东': '宜昌', '温州南': '温州', '自贡': '自贡', '海口东': '海口', '济南东': '济', '嘉兴南': '嘉兴', '张家界西': '张家界', '南昌西': '南昌', '利川': '利川', '沈阳北': '沈阳', '徐州东': '徐州', '宁波': '宁波', '武昌': '武昌', '汕头南': '汕头', '美兰': '美兰', '赣州': '赣州', '连云港东': '连云港', '洛阳': '洛阳', '青岛北': '青岛', '汕头': '汕头', '兰州': '兰州', '上海': '上海', '安阳东': '安阳', '六盘水': '六盘水', '南昌南': '南昌', '运城北': '运城', '福州南': '福州', '江油': '江油', '苏州': '苏州', '贵阳': '贵阳', '防城港北': '防城港', '衢州': '衢州', '淮安东': '淮安', '芜湖': '芜湖', '汉中': '汉中', '邛崃': '邛崃', '安庆': '安庆', '牡丹江': '牡丹江', '攀枝花南': '攀枝花', '老城镇': '老城镇', '长春': '长春', '巴中东': '巴中', '深圳': '深圳', '桂林北': '桂林', '吉首东': '吉首', '厦门北': '厦门', '天津': '天津', '邯郸': '邯郸', '杭州东': '杭州', '瑞安': '瑞安', '铜仁': '铜仁', '邵阳': '邵阳', '杭州': '杭州', '百色': '百色', '桂林': '桂林', '佳木斯': '佳木斯', '齐齐哈尔': '齐齐哈尔', '海  口东': '海  口', '大连': '大连', '西宁': '西宁', '临沂北': '临沂', '阳新': '阳新', '南昌': '南昌', '万源': '万源', '嵊州新昌': '嵊州新昌', '绥化': '绥化', '太仓': '太仓', '南通': '南通', '潼南': '潼', '台州': '台州', '涪陵北': '涪陵', '西安北': '西安', '郑州东': '郑州', '潍坊': '潍坊', '东莞南': '东莞', '湛江': '湛江', '长春西': '长春', '菏泽东': '菏泽', '东方': '东方', '大同': '大同', '十堰': '十堰', '平潭': '平潭', '青城山': '青城山', '北京': '北京', '海口': '海口', '乌鲁木齐': '乌鲁木齐', '周口': '周口', '哈尔滨西': '哈尔滨', '山海关': '山海关', '阜阳西': '阜阳', '包头': '包头', '广安南': '广安', '潮汕': '潮汕', '黄冈东': '黄冈', '叙永北': '叙永', '深圳北': '深圳', '贵阳北': '贵阳', '昆明南': '昆明', '襄阳东': '襄阳', '上饶': '上饶', '临汾': '临汾', '高兴': '高兴', '梅州西': '梅州', '柳州': '柳州', '宁德': '宁德', '永州': '永州', '十堰东': '十堰', '龙岩': '龙岩', '乌兰浩特': '乌兰浩特', '淮北': '淮', '淄博北': '淄博', '郑州': '郑州', '南京南': '南京', '莘县': '莘县', '太原': '太原', '南 昌': '南 昌', '济  南西': '济', '长沙': '长沙', '萍乡': '萍乡', '三门县': '三门县', '六安': '六安', '南宁东': '南宁', '丹东': '丹', '北京西': '北京', '青岛': '青岛', '临  沂北': '临  沂', '毕节': '毕节', '呼和浩特东': '呼和浩特', '岳阳东': '岳阳', '深圳东': '深圳', '上海南': '上海', '太原南': '太原', '台州西': '台州', '延安': '延安', '重庆西': '重庆', '福鼎': '福鼎', '呼和浩特': '呼和浩特', '上海虹桥': '上海虹桥', '广州白云': '广州白云', '饶平': '饶平', '三亚': '三亚', '福田': '福田', '绥芬河': '绥芬河', '敦煌': '敦煌', '宜宾': '宜宾', '绵阳': '绵阳', '双鸭山西': '双鸭山', '东莞东': '东莞', '武胜': '武胜', '惠州': '惠州', '襄阳': '襄阳', '达州': '达州', '平阳': '平阳', '阜阳': '阜阳', '张家港': '张家港', '惠州北': '惠州', '兰州西': '兰州', '黄山北': '黄山', '南昌东': '南昌', '威海': '威海', '北京丰台': '北京丰台', '武汉': '武汉', '赣榆': '赣榆', '广元': '广元', '广州': '广州', '昆山': '昆山', '香港西九龙': '香港西九龙', '石家庄': '石家庄', '宿松东': '宿松', '佛山西': '佛山'
}
# 计算耗时（分钟）
def calculate_travel_time(row):
    start_time_str = row['start_time']
    arrive_time_str = row['arrive_time']

    # 将24:00转换为00:00
    if start_time_str == '24:00':
        start_time_str = '00:00'
    if arrive_time_str == '24:00':
        arrive_time_str = '00:00'

    start_time = datetime.strptime(start_time_str, '%H:%M')
    arrive_time = datetime.strptime(arrive_time_str, '%H:%M')

    # 处理跨天情况
    if arrive_time < start_time:
        arrive_time += timedelta(days=1)

    travel_time = (arrive_time - start_time).total_seconds() / 60
    return travel_time


train_df['travel_time'] = train_df.apply(calculate_travel_time, axis=1)

# 处理票价中的异常值
train_df['wz_price'] = train_df['wz_price'].replace('--', '0').astype(float)

# 创建城市之间的高铁网络图
G = nx.Graph()

for index, row in train_df.iterrows():
    G.add_edge(row['from_station_name'], row['end_station_name'], weight=row['travel_time'], cost=row['wz_price'])

# 将城市替换为高铁站
train_df['from_city'] = train_df[start_station_col].map(city_to_station)
train_df['to_city'] = train_df[end_station_col].map(city_to_station)

# 选择最令外国游客向往的50个城市（假设已经有这个名单）
top_50_cities = ['北京', '重庆', '广州', '成都', '天津',
                 '西安', '南京', '武汉', '苏州', '深圳',
                 '杭州', '长沙', '哈尔滨', '郑州', '合肥',
                 '贵阳', '济南', '沈阳', '昆明', '晋中',
                 '福州', '长春', '常德', '衡阳', '大连',
                 '南昌', '宁波', '南宁', '石家庄', '太原',
                 '温州', '无锡', '呼伦贝尔', '遵义', '青岛',
                 '泉州', '佛山', '南通', '徐州', '洛阳',
                 '新余', '烟台', '常州', '呼和浩特', '赣州',
                 '厦门', '潍坊', '宜春', '镇江', '台州'
                 ]  # 用实际城市名称替代

# 从每个城市选择评分最高的景点
top_sights = city_sights_df.loc[city_sights_df.groupby('城市')['评分'].idxmax()]

# 过滤出50个城市的景点
filtered_sights = top_sights[top_sights['城市'].isin(top_50_cities)]

# 计算每个城市的总游玩时间和总费用
def travel_cost_and_time(path, filtered_sights, G):
    total_time = 0
    total_cost = 0
    for i in range(len(path) - 1):
        total_time += G[path[i]][path[i + 1]]['weight']
        total_cost += G[path[i]][path[i + 1]]['cost']
    # 加上景点游玩的时间和门票
    for station in path:
        city = next(key for key, value in city_to_station.items() if value == station)
        sight = filtered_sights[filtered_sights['城市'] == city].iloc[0]
        total_time += sight['建议游玩时长']
        total_cost += sight['门票']
    return total_time, total_cost

# 使用近似算法解决TSP问题
from itertools import permutations


def tsp_approximate(G, start, filtered_sights):
    stations = list(filtered_sights['城市'].unique())
    min_path = None
    min_cost = float('inf')
    min_time = float('inf')
    for perm in permutations(stations):
        perm = (start,) + perm
        time, cost = travel_cost_and_time(perm, filtered_sights, G)
        if cost < min_cost:
            min_cost = cost
            min_time = time
            min_path = perm
    return min_path, min_time, min_cost


# 假设从广州南开始旅行
start_station = city_to_station['广州']
path, total_time, total_cost = tsp_approximate(G, start_station, filtered_sights)

# 输出结果
print(f"规划的游玩路线: {path}")
print(f"总花费时间: {total_time} 分钟")
print(f"门票和交通的总费用: {total_cost} 元")

# 可视化路线
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="r", width=2)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.show()


