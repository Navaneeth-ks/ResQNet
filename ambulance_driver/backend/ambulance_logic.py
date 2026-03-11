import math

# ══════════════════════════════════════════════════
#  AMBULANCE FLEET
#  Update names/positions to match your real fleet
# ══════════════════════════════════════════════════
fleet = [
    {
        "id":       "AMB-A",
        "name":     "Ambulance A",
        "driver":   "John",
        "phone":    "9876543210",
        "position": {"lat": 9.9930419, "lng": 76.3017048},
        "status":   "available"
    },
    {
        "id":       "AMB-B",
        "name":     "Ambulance B",
        "driver":   "Suresh M.",
        "phone":    "9876543211",
        "position": {"lat": 9.9980000, "lng": 76.2990000},
        "status":   "available"
    },
    {
        "id":       "AMB-C",
        "name":     "Ambulance C",
        "driver":   "Priya N.",
        "phone":    "9876543212",
        "position": {"lat": 9.9870000, "lng": 76.3050000},
        "status":   "available"
    },
    {
        "id":       "AMB-D",
        "name":     "Ambulance D",
        "driver":   "Arun V.",
        "phone":    "9876543213",
        "position": {"lat": 9.9850000, "lng": 76.2850000},
        "status":   "offline"
    },
]


# ══════════════════════════════════════════════════
#  HAVERSINE — distance in metres between 2 coords
# ══════════════════════════════════════════════════
def haversine(lat1, lng1, lat2, lng2):
    R     = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a     = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(d_lng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ══════════════════════════════════════════════════
#  FIND NEAREST available ambulance
# ══════════════════════════════════════════════════
def find_nearest(accident_lat: float, accident_lng: float):
    nearest  = None
    min_dist = float("inf")

    for amb in fleet:
        if amb["status"] != "available":
            continue
        d = haversine(
            amb["position"]["lat"],
            amb["position"]["lng"],
            accident_lat,
            accident_lng
        )
        if d < min_dist:
            min_dist = d
            nearest  = amb

    return nearest, round(min_dist)


# ══════════════════════════════════════════════════
#  FORMAT distance → "250m" or "1.25km"
# ══════════════════════════════════════════════════
def format_distance(metres: int) -> str:
    if metres < 1000:
        return f"{metres}m"
    return f"{metres / 1000:.2f}km"


# ══════════════════════════════════════════════════
#  GET single ambulance by ID
# ══════════════════════════════════════════════════
def get_ambulance(amb_id: str):
    return next((a for a in fleet if a["id"] == amb_id), None)


# ══════════════════════════════════════════════════
#  GET single ambulance by NAME
#  (matches what your friend writes to Firestore
#   e.g. "Ambulance A")
# ══════════════════════════════════════════════════
def get_ambulance_by_name(name: str):
    return next((a for a in fleet if a["name"] == name), None)
