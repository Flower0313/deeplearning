import re

import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    passwd='root',
    charset='utf8mb4'
)
cursor = conn.cursor()

sql = '''
SELECT channel_id,message_id,group_id,remark FROM spider_base.`ods_building_phoenix` where channel_id='1552083977'
'''

insert_sql = '''
insert into spider_base.`dwd_building_phoenix`(channel_id,message_id,group_id,province,city,area,if_sw,if_by,if_door,if_96,remark,min_price,real_content,age,city_id,height,weight) values("{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","3100","{}","{}")
'''

cursor.execute(sql)
content = cursor.fetchall()
qq = r"(?<=[q|Q|Qq|qq|QQ].{1}[:：]{1}?).+\d+"
tel = r"(?<!\d)(?:1[3456789]\d{9})(?!\d)"
wx = r"(?<=([微信|wx|v|V|微|vx|VX|Vx|Wx|WX]{1}[:：]{1})).+[\w\-\_]{6,20}"
tele = r"@[a-zA-Z0-9_]*"
contact = r"(?<=联系方式[:：]{1})[a-zA-Z0-9_|，]+|[a-zA-Z0-9_]{6,20}"

pattern = re.compile(qq + '|' + tel + '|' + wx + '|' + tele + '|' + contact + '|' + r"\b[a-zA-Z0-9_]{6,20}\b")
for i in content:
    # 去除名字
    origin = re.sub(r"编号：|https:\/\/t\.me\/c\/\d+\/\d+", '', str(i[3]))

    age = []
    # 隐藏联系方式(简介)
    result = pattern.sub('已隐藏', origin)
    # 获取水费
    money = re.findall(r"\d{1,5}(?=P|p)|\d{1,5}(?=/)", result)
    money = 0 if len(money) == 0 else min(int(x) for x in money)

    # 获取身高
    height = re.findall(r"\b1\d{2}\b", result)
    height = 0 if len(height) == 0 else min(int(x) for x in height)

    # 获取体重
    weight = re.findall(r"\b\d.(?=kg|KG)", result)
    weight = 0 if len(weight) == 0 else min(int(x) for x in weight)

    # 获取区域
    area = re.findall(r"黄浦|徐汇|长宁|静安|普陀|虹口|杨浦|闵行|宝山|嘉定|浦东新|金山|松江|青浦|奉贤|崇明", result)
    # 是否by
    if_by = 1 if re.search(r'[夜|by|包夜]', result) is not None else 0
    # 是否上门
    if_door = 1 if result.__contains__("上门") else 0
    # 是否sw
    if_sw = 1 if result.__contains__("sw") else 0
    # 是否96
    if_96 = 1 if re.search(r"\b69\b", result) is not None else 0

    area = '无' if not area else area

    e_sql = insert_sql.format(i[0], i[1], i[2], '上海', '上海市', str(area[0]) + '区', if_sw, if_by, if_door, if_96,
                              str(result), money,
                              origin, 0 if len(age) == 0 else age[0], height, weight)
    # print(e_sql)
    cursor.execute(e_sql)
    conn.commit()

cursor.close()
conn.close()
