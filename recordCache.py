import os, pickle, sys
from cache import LFUCache


cacheName = sys.argv[1]
dirname = os.path.dirname(__file__)
capacity = 400
lfuCachePath = os.path.join(dirname, "%s.txt"%cacheName)

cache = LFUCache(capacity)
print(cache)
with open(lfuCachePath, "wb") as f:
	pickle.dump(cache, f)