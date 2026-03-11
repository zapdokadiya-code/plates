import unicodedata
import string

weird_text = "ＡＢＣ𝔻𝔼𝔽ＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
normalized = unicodedata.normalize('NFKD', weird_text).encode('ascii', 'ignore').decode('ascii')
print(f"Original: {weird_text}")
print(f"Normalized: {normalized}")

devanagari = "एम एच"
print(f"Devanagari: {unicodedata.normalize('NFKD', devanagari).encode('ascii', 'ignore').decode('ascii')}")
