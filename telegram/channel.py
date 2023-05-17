import asyncio
import datetime

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
    host='127.0.0.1',
    port=3306,
    user='root',
    passwd='root',
    charset='utf8mb4'
)
cursor = conn.cursor()

base_sql_zero_kline = '''
    insert into spider_base.`ods_building_phoenix`(`channel_name`,`channel_id`,`message_id`,`group_id`,`remark`) value("{}","{}","{}","{}","{}")
'''

client = TelegramClient('fucker', api_id, api_hash, proxy=proxy)


# 如果有grouped_id就以这个为主，grouped_id为None就以id为主
def main():
    client.start()
    # 频道实体
    entity = client.get_entity(Channel(id=1552083977, title='上海已验证资源（长三角群）',
                                       photo=ChatPhoto(photo_id=6262654211762269595, dc_id=5, has_video=False,
                                                       stripped_thumb=''),
                                       date=datetime.datetime(2023, 4, 12, 10, 34, 37, tzinfo=datetime.timezone.utc),
                                       creator=False, left=False, broadcast=True, verified=False, megagroup=False,
                                       restricted=False, signatures=False, min=False, scam=False, has_link=False,
                                       has_geo=False, slowmode_enabled=False, call_active=False, call_not_empty=False,
                                       fake=False, gigagroup=False, noforwards=False, join_to_send=False,
                                       join_request=False, forum=False, access_hash=2422350542710402032, username=None,
                                       restriction_reason=[], admin_rights=None, banned_rights=None,
                                       default_banned_rights=None, participants_count=2832, usernames=[]))

    # 获取频道的消息
    messages = client.iter_messages(entity, min_id=1)

    for message in messages:
        # 获取媒体对象
        media = message.media
        # 图片
        if message.photo:
            # 保存图片
            # message.download_media(
            #     file=r'T:\deeplearning\imgs\\' + str(message.peer_id.channel_id) + '-' + str(message.id) + '-' + str(
            #         message.grouped_id) + '.jpg')
            if str(message.message) != '' and message.message is not None:
                group_id = 0 if message.grouped_id is None else str(message.grouped_id)
                e_sql = base_sql_zero_kline.format(str(utils.get_display_name(message.sender)),
                                                   str(message.peer_id.channel_id), str(message.id), group_id,
                                                   str(message.message))
                print(e_sql)
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
            print(e_sql)
            cursor.execute(e_sql)
            conn.commit()

            # 视频
            if media is not None and isinstance(media, MessageMediaDocument) and media.document.mime_type.startswith(
                    'video/'):
                print(1)
                # message.download_media(
                #     file=r'T:\deeplearning\imgs\\' + str(message.peer_id.channel_id) + '-' + str(
                #         message.id) + '-' + str(
                #         message.grouped_id) + '.MP4')


with client:
    try:
        main()
    except asyncio.CancelledError:
        print("任务被取消")
