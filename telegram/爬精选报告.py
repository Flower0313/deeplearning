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
    entity = client.get_entity(Channel(id=1701733473, title='上海精选榜单报告',
                                       photo=ChatPhoto(photo_id=4978733756496063088, dc_id=1, has_video=False,
                                                       stripped_thumb=''),
                                       date=datetime.datetime(2023, 5, 30, 7, 28, 25, tzinfo=datetime.timezone.utc),
                                       creator=False, left=False, broadcast=True, verified=False, megagroup=False,
                                       restricted=False, signatures=False, min=False, scam=False, has_link=True,
                                       has_geo=False, slowmode_enabled=False, call_active=False, call_not_empty=False,
                                       fake=False, gigagroup=False, noforwards=False, join_to_send=False,
                                       join_request=False, forum=False, access_hash=-1629641722016354516,
                                       username='tgxbd', restriction_reason=[], admin_rights=None, banned_rights=None,
                                       default_banned_rights=None, participants_count=29320, usernames=[]))

    # 获取频道的消息
    messages = client.iter_messages(entity, min_id=1)

    for message in messages:
        # 获取媒体对象
        media = message.media
        # 图片
        if message.photo:
            # 保存图片 频道-消息-组
            # message.download_media(
            #     file=r'T:\deeplearning\imgs\\' + str(message.peer_id.channel_id) + '-' + str(message.id) + '-' + str(
            #         0 if message.grouped_id is None else message.grouped_id) + '.jpg')
            if str(message.message) != '' and message.message is not None and "已下架" not in str(message.message) and "SPA" not in str(message.message) and str(message.message).__contains__("@"):
                group_id = 0 if message.grouped_id is None else str(message.grouped_id)
                e_sql = base_sql_zero_kline.format(str(utils.get_display_name(message.sender)),
                                                   str(message.peer_id.channel_id), str(message.id), group_id,
                                                   str(message.message).replace("\n", ""))
                #print(e_sql)
                cursor.execute(e_sql)
                conn.commit()


with client:
    try:
        main()
    except asyncio.CancelledError:
        print("任务被取消")
