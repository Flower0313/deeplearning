import asyncio

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


# async def main():
#     await client.start()
#
#     # 指定要下载数据的频道链接
#     channel_link = 'https://t.me/+CuRuJByI1uxmYTRl'
#
#     # 下载频道消息
#     messages = client.iter_messages(channel_link, limit=100)
#     async for message in messages:
#         msg = str(message.date) + '[' + str(utils.get_display_name(message.sender)) + ':' + str(message.message) + ']\n'
#         print(msg)
#
#
# with client:
#     client.loop.run_until_complete(main())

# 如果有grouped_id就以这个为主，grouped_id为None就以id为主
def main():
    client.start()

    # 指定要下载数据的频道链接
    channel_link = 'https://t.me/testdemo003'  # 'https://t.me/shanghaierlang1'
    # 下载频道消息
    # 频道名称：str(utils.get_display_name(message.sender))
    messages = client.iter_messages(channel_link, min_id=1, max_id=100)
    for message in messages:
        # client.download_media(message.media,'T:\deeplearning\imgs')
        media = message.media
        # 图片
        if message.photo:
            # 保存图片
            message.download_media(
                file=r'T:\deeplearning\imgs\\' + str(message.id) + '-' + str(message.grouped_id) + '.jpg')
            if str(message.message) != '' and message.message is not None:
                print('聊天ID:' + str(message.id) + '-' + str(message.grouped_id) +
                      '\n文本消息:' + str(message.message) + '\n --------------------')

        # 文本
        elif message.text:
            msg = '聊天ID:' + str(message.id) + '-' + str(message.grouped_id) \
                  + '\n文本消息:' + str(message.message) + '\n--------------------'
            print(msg)
            # 视频
            if media is not None and isinstance(media, MessageMediaDocument) and media.document.mime_type.startswith(
                    'video/'):
                message.download_media(
                    file=r'T:\deeplearning\imgs\\' + str(message.id) + '-' + str(message.grouped_id) + '.MP4')


with client:
    try:
        main()
    except asyncio.CancelledError:
        print("任务被取消")
