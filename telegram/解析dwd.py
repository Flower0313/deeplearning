import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    passwd='root',
    charset='utf8mb4'
)
cursor = conn.cursor()

# 上海区正则：黄浦区|徐汇区|长宁区|静安区|普陀区|虹口区|杨浦区|闵行区|宝山区|嘉定区|浦东新区|金山区|松江区|青浦区|奉贤区|崇明区
# 消除SXX：S\d*(?=\,).
# 获取年龄数字： (?<![a-zA-Z0-9])\d{2}(?!\d)
# 获取身高：(?<!\d)1\d{2}(?!\d)
# 获取QQ：(?<=[qQ]+[:：]?)\d+
# 获取手机号码：(?<!\d)(?:1[3456789]\d{9})(?!\d)
# 获取微信：(?<=((微信|wx|v|V|微|vx|VX|Vx|Wx|WX)+[:：])).+[\w\-\_]{6,20}
sql = '''
SELECT remark FROM spider_base.`ods_building_phoenix` where channel_id='1708774228' limit 10
'''

cursor.execute(sql)
content = cursor.fetchall()
for i in content:
    print(i)
