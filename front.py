import re

text = '''
Channel(id=1596029069, title='长三角电报总群', photo=ChatPhoto(photo_id=6206058676083601764, dc_id=5, has_video=False, stripped_thumb=b'\x01\x08\x08\x86\x1b\xb8\xf1\x1a\x90A\x04\x0c\x9e\xde\xb4QE+\x0e\xe7'), date=date
'''

new_text = re.sub(r"(?<=stripped_thumb=).*?(?=\),)", "''", text)
print(new_text)
