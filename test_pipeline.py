import logging
logging.basicConfig(level=logging.DEBUG)

from ai.preprocessing.text_normalizer import TextNormalizer
normalizer = TextNormalizer()
text = "enaku romba kastama iruku"
print("Input:", text)
output = normalizer.normalize(text)
print("Output:", output)
