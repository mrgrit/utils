import heapq

heap = []

heapq.heappush(heap, (1,'one'))
heapq.heappush(heap, (10,'ten'))
heapq.heappush(heap, (5,'five'))


for x in heap:
    print(x)

heapq.heappop(heap)

for x in heap:
    print(x)
