import heapq

a1 = (5, (2,3))
a2 = (6, (2,4))
a3 = (-1, (2,4))
a4 = (5, (2,3))


heap = []
heapq.heappush(heap, a1)
print(heap)
heapq.heappush(heap, a2)
print(heap)
heapq.heappush(heap, a3)
print(heap)
heapq.heappush(heap, a4)
print(heap)
print(heapq.heappop(heap))
print(heapq.heappop(heap))
print(heapq.heappop(heap))
print(heapq.heappop(heap))
