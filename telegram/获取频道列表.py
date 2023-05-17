import asyncio
import datetime
import re

import pymysql
from telethon.sync import TelegramClient
from telethon import utils
from telethon import types
from telethon.tl.types import MessageMediaDocument, Channel, ChatPhoto

api_id = 22434812
api_hash = '878956832216ec65fd5f9814483b02d9'
proxy = {
    'proxy_type': 'socks5',
    'addr': '127.0.0.1',
    'port': 7890
}

conn = pymysql.connect(
    host='8.130.49.69',
    port=3306,
    user='root',
    passwd='w654646',
    charset='utf8mb4'
)
cursor = conn.cursor()

base_sql_zero_kline = '''
    insert into spider_base.`channel_phoenix`(`channel_id`,`channel_entity`) value("{}","{}")
'''

# 创建TelegramClient对象并登录
client = TelegramClient('fucker', api_id, api_hash, proxy=proxy)
client.start()
# 获取当前登录用户所关注的所有频道
channels = client.get_dialogs()
for channel in channels:
    if channel.is_channel:  # 确定是否为频道
        e_sql = base_sql_zero_kline.format(channel.entity.id, re.sub(r"(?<=stripped_thumb=).*?(?=\),)", "''", str(channel.entity)))
        print(e_sql)
        # cursor.execute(e_sql)
        # conn.commit()

cursor.close()
conn.close()