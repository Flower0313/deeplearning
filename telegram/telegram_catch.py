import asyncio

import pymysql
from telethon.sync import TelegramClient
from telethon import utils
from telethon import types
from telethon.tl.types import MessageMediaDocument

api_id = 22434812
api_hash = '878956832216ec65fd5f9814483b02d9'
proxy = {
    'proxy_type': 'socks5',
    'addr': '127.0.0.1',
    'port': 7890
}

client = TelegramClient('fucker', api_id, api_hash, proxy=proxy)
conn = pymysql.connect(
    host='8.130.49.69',
    port=3306,
    user='root',
    passwd='w654646',
    charset='utf8'
)
cursor = conn.cursor()

base_sql_zero_kline = '''
    insert into spider_base.`building_phoenix`(`channel_name`,`channel_id`,`message_id`,`group_id`,`remark`) value("{}","{}","{}","{}","{}")
'''


# 如果有grouped_id就以这个为主，grouped_id为None就以id为主
def main():
    client.start()

    # 指定要下载数据的频道链接
    channel_link = 'https://t.me/testdemo003'  # 'https://t.me/+I5znaD5309RjYWM1'
    # 下载频道消息
    messages = client.iter_messages(channel_link, min_id=1, max_id=100)
    for message in messages:
        # 获取媒体对象
        media = message.media
        # 图片
        if message.photo:
            # 保存图片
            message.download_media(
                file=r'T:\deeplearning\imgs\\' + str(message.peer_id.channel_id) + '-' + str(message.id) + '-' + str(
                    message.grouped_id) + '.jpg')
            if str(message.message) != '' and message.message is not None:
                group_id = 0 if message.grouped_id is None else str(message.grouped_id)
                e_sql = base_sql_zero_kline.format(str(utils.get_display_name(message.sender)),
                                                   str(message.peer_id.channel_id), str(message.id), group_id,
                                                   str(message.message))
                cursor.execute(e_sql)
                conn.commit()
        # 文本
        elif message.text:
            msg = '聊天ID:' + str(message.peer_id.channel_id) + '-' + str(message.id) + '-' + str(message.grouped_id) \
                  + '\n文本消息:' + str(message.message) + '\n--------------------'
            group_id = 0 if message.grouped_id is None else str(message.grouped_id)
            e_sql = base_sql_zero_kline.format(str(utils.get_display_name(message.sender)),
                                               str(message.peer_id.channel_id), str(message.id), group_id,
                                               str(message.message))
            cursor.execute(e_sql)
            conn.commit()

            # 视频
            if media is not None and isinstance(media, MessageMediaDocument) and media.document.mime_type.startswith(
                    'video/'):
                message.download_media(
                    file=r'T:\deeplearning\imgs\\' + str(message.peer_id.channel_id) + '-' + str(
                        message.id) + '-' + str(
                        message.grouped_id) + '.MP4')


with client:
    try:
        main()
    except asyncio.CancelledError:
        print("任务被取消")
    finally:
        cursor.close()
        conn.close()
