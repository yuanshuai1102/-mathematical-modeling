import pandas as pd
import numpy as np

# 读取高铁路线数据
train_data = pd.read_csv(r"train (2).csv")

# 假设列名如下，替换为实际列名
start_station_col = 'start_station_name'
end_station_col = 'end_station_name'
start_time_col = 'start_time'
end_time_col = 'arrive_time'
cost_col = 'zy_price'

# 站点到城市的映射字典
station_to_city = {'广州南': '广州', '北京南': '北京', '重庆北': '重庆', '成都东': '成都', '天津西': '天津', '玉门': '玉门', '吉林': '吉林', '梧州南': '梧州', '南平市': '南平市', '济南西': '济', '巴东': '巴', '南宁': '南宁', '峨眉': '峨眉', '厦门': '厦门', '通化': '通化', '沈阳': '沈阳', '昆明': '昆明', '惠阳': '惠阳', '太湖南': '太湖', '北海': '北海', '宁海': '宁海', '常州北': '常州', '赣州西': '赣州', '烟台': '烟台', '盐城': '盐城', '汉口': '汉口', '咸宁南': '咸宁', '长沙南': '长沙', '湛江西': '湛江', '通辽': '通辽', '徐州': '徐州', '济南': '济', '大连北': '大连', '齐齐哈尔南': '齐齐哈尔', '株洲西': '株洲', '丽水': '丽水', '西安': '西安', '苍南': '苍', '南京': '南京', '杭州西': '杭州', '石柱县': '石柱县', '广州东': '广州', '嘉峪关南': '嘉峪关', '九江': '九江', '宜兴': '宜兴', '泰宁': '泰宁', '银川': '银川', '福州': '福州', '温州': '温州', '荣成': '荣成', '三  亚': '三  亚', '合肥南': '合肥', '曲阜东': '曲阜', '延吉西': '延吉', '盘州': '盘州', '珠海': '珠海', '蚌埠南': '蚌埠', '成都西': '成都', '亳州南': '亳州', '宜昌东': '宜昌', '温州南': '温州', '自贡': '自贡', '海口东': '海口', '济南东': '济', '嘉兴南': '嘉兴', '张家界西': '张家界', '南昌西': '南昌', '利川': '利川', '沈阳北': '沈阳', '徐州东': '徐州', '宁波': '宁波', '武昌': '武昌', '汕头南': '汕头', '美兰': '美兰', '赣州': '赣州', '连云港东': '连云港', '洛阳': '洛阳', '青岛北': '青岛', '汕头': '汕头', '兰州': '兰州', '上海': '上海', '安阳东': '安阳', '六盘水': '六盘水', '南昌南': '南昌', '运城北': '运城', '福州南': '福州', '江油': '江油', '苏州': '苏州', '贵阳': '贵阳', '防城港北': '防城港', '衢州': '衢州', '淮安东': '淮安', '芜湖': '芜湖', '汉中': '汉中', '邛崃': '邛崃', '安庆': '安庆', '牡丹江': '牡丹江', '攀枝花南': '攀枝花', '老城镇': '老城镇', '长春': '长春', '巴中东': '巴中', '深圳': '深圳', '桂林北': '桂林', '吉首东': '吉首', '厦门北': '厦门', '天津': '天津', '邯郸': '邯郸', '杭州东': '杭州', '瑞安': '瑞安', '铜仁': '铜仁', '邵阳': '邵阳', '杭州': '杭州', '百色': '百色', '桂林': '桂林', '佳木斯': '佳木斯', '齐齐哈尔': '齐齐哈尔', '海  口东': '海  口', '大连': '大连', '西宁': '西宁', '临沂北': '临沂', '阳新': '阳新', '南昌': '南昌', '万源': '万源', '嵊州新昌': '嵊州新昌', '绥化': '绥化', '太仓': '太仓', '南通': '南通', '潼南': '潼', '台州': '台州', '涪陵北': '涪陵', '西安北': '西安', '郑州东': '郑州', '潍坊': '潍坊', '东莞南': '东莞', '湛江': '湛江', '长春西': '长春', '菏泽东': '菏泽', '东方': '东方', '大同': '大同', '十堰': '十堰', '平潭': '平潭', '青城山': '青城山', '北京': '北京', '海口': '海口', '乌鲁木齐': '乌鲁木齐', '周口': '周口', '哈尔滨西': '哈尔滨', '山海关': '山海关', '阜阳西': '阜阳', '包头': '包头', '广安南': '广安', '潮汕': '潮汕', '黄冈东': '黄冈', '叙永北': '叙永', '深圳北': '深圳', '贵阳北': '贵阳', '昆明南': '昆明', '襄阳东': '襄阳', '上饶': '上饶', '临汾': '临汾', '高兴': '高兴', '梅州西': '梅州', '柳州': '柳州', '宁德': '宁德', '永州': '永州', '十堰东': '十堰', '龙岩': '龙岩', '乌兰浩特': '乌兰浩特', '淮北': '淮', '淄博北': '淄博', '郑州': '郑州', '南京南': '南京', '莘县': '莘县', '太原': '太原', '南 昌': '南 昌', '济  南西': '济', '长沙': '长沙', '萍乡': '萍乡', '三门县': '三门县', '六安': '六安', '南宁东': '南宁', '丹东': '丹', '北京西': '北京', '青岛': '青岛', '临  沂北': '临  沂', '毕节': '毕节', '呼和浩特东': '呼和浩特', '岳阳东': '岳阳', '深圳东': '深圳', '上海南': '上海', '太原南': '太原', '台州西': '台州', '延安': '延安', '重庆西': '重庆', '福鼎': '福鼎', '呼和浩特': '呼和浩特', '上海虹桥': '上海虹桥', '广州白云': '广州白云', '饶平': '饶平', '三亚': '三亚', '福田': '福田', '绥芬河': '绥芬河', '敦煌': '敦煌', '宜宾': '宜宾', '绵阳': '绵阳', '双鸭山西': '双鸭山', '东莞东': '东莞', '武胜': '武胜', '惠州': '惠州', '襄阳': '襄阳', '达州': '达州', '平阳': '平阳', '阜阳': '阜阳', '张家港': '张家港', '惠州北': '惠州', '兰州西': '兰州', '黄山北': '黄山', '南昌东': '南昌', '威海': '威海', '北京丰台': '北京丰台', '武汉': '武汉', '赣榆': '赣榆', '广元': '广元', '广州': '广州', '昆山': '昆山', '香港西九龙': '香港西九龙', '石家庄': '石家庄', '宿松东': '宿松', '佛山西': '佛山'}

# 预处理时间数据
def preprocess_time(time_str):
    if time_str == '24:00':
        return '00:00'
    return time_str

train_data[start_time_col] = train_data[start_time_col].apply(preprocess_time)
train_data[end_time_col] = train_data[end_time_col].apply(preprocess_time)

# 将具体位置转换为城市名称
train_data['from_city'] = train_data[start_station_col].map(station_to_city)
train_data['to_city'] = train_data[end_station_col].map(station_to_city)

# 将费用列转换为数值格式
train_data[cost_col] = pd.to_numeric(train_data[cost_col], errors='coerce').fillna(0)

# 计算行程时间函数
def calculate_travel_time(row):
    start_time = pd.to_datetime(row[start_time_col])
    end_time = pd.to_datetime(row[end_time_col])
    # 考虑到可能跨午夜的情况
    if end_time < start_time:
        end_time += pd.Timedelta(days=1)
    travel_time = (end_time - start_time).total_seconds() / 3600  # 转换为小时
    return travel_time

# 添加行程时间列
train_data['travel_time'] = train_data.apply(calculate_travel_time, axis=1)

# 打印处理后的高铁数据以确认
print(train_data.head())

# 获取50个最令外国游客向往的城市，去掉“市”字
top_50_cities = [
    '北京', '重庆', '广州', '成都', '天津',
    '西安', '南京', '武汉', '苏州', '深圳',
    '杭州', '长沙', '哈尔滨', '郑州', '合肥',
    '贵阳', '济南', '沈阳', '昆明', '晋中',
    '福州', '长春', '常德', '衡阳', '大连',
    '南昌', '宁波', '南宁', '石家庄', '太原',
    '温州', '无锡', '呼伦贝尔', '遵义', '青岛',
    '泉州', '佛山', '南通', '徐州', '洛阳',
    '新余', '烟台', '常州', '呼和浩特', '赣州',
    '厦门', '潍坊', '宜春', '镇江', '台州'
]
start_city = '广州'
time_limit = 144  # 小时

# 定义函数计算城市之间的高铁时间和费用
def get_train_info(city1, city2, train_data):
    route = train_data[(train_data['from_city'] == city1) & (train_data['to_city'] == city2)]
    print(f"Checking route from {city1} to {city2}: {len(route)} found")  # 添加调试信息
    if route.empty:
        return float('inf'), float('inf')
    travel_time = route.iloc[0]['travel_time']
    travel_cost = route.iloc[0][cost_col]
    return travel_time, travel_cost

# 规划游玩路线的贪心算法
def plan_route_greedy_cost_optimized(start_city, time_limit, top_50_cities, train_data):
    current_time = 0
    current_city = start_city
    total_cost = 0
    visited_cities = [start_city]

    while current_time < time_limit:
        next_city = None
        min_cost_per_time = float('inf')

        # 找到下一个可以访问的城市
        for city in top_50_cities:
            if city not in visited_cities:
                travel_time, travel_cost = get_train_info(current_city, city, train_data)
                print(f"From {current_city} to {city}: time {travel_time}, cost {travel_cost}")  # 添加调试信息
                cost_per_time = travel_cost / travel_time if travel_time > 0 else float('inf')
                if cost_per_time < min_cost_per_time and current_time + travel_time + 2 < time_limit:  # 假设每个景点游览时间为2小时
                    min_cost_per_time = cost_per_time
                    next_city = city
                    next_cost = travel_cost

        if next_city is None:
            break

        # 更新行程信息
        current_time += min_cost_per_time + 2  # 包括游览时间
        total_cost += next_cost
        visited_cities.append(next_city)
        current_city = next_city

    return visited_cities, current_time, total_cost

# 执行路线规划
visited_cities, total_time, total_cost = plan_route_greedy_cost_optimized(start_city, time_limit, top_50_cities, train_data)

# 输出结果
print("访问的城市:", visited_cities)
print("总花费时间:", total_time)
print("总费用:", total_cost)
