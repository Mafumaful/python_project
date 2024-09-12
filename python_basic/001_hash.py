class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p1 = Point(1, 2)
p2 = Point(1, 2)

# Hash values
print(hash(p1))
print(hash(p2))

point_dict = {p1: "Point 1"}

# Since p2 has the same coordinates as p1, it's treated as the same key
print(point_dict[p2])  # Output: "Point 1"
