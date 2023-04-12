from telethon.sync import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos, InputMessagesFilterDocument
from telethon import utils

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


def main():
    client.start()

    # 指定要下载数据的频道链接 #
    channel_link = 'https://t.me/+kO8OD8i4-Xg0Yzdl'  # https://t.me/+CuRuJByI1uxmYTRl

    # 下载频道消息
    messages = client.iter_messages(channel_link, limit=100)
    for message in messages:
        msg = str(message.date) + '[' + str(utils.get_display_name(message.sender)) + ':' + str(message.message) + ']\n'
        print(msg)


with client:
    main()
