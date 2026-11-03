"""The vocabulary a registry record is built from.

Plausible Indian makes, models, RTO offices, insurers and case types, grouped
by the vehicle classes the platform's simulator actually emits (car,
two_wheeler, auto, truck, taxi, bus). Nothing here is scraped or licensed - it
is a hand-written vocabulary chosen so a record reads like a real one.
"""
from __future__ import annotations

# --- vehicles by class --------------------------------------------------
MODELS: dict[str, list[tuple[str, str, list[str]]]] = {
    # class: [(make, model, [fuels])]
    "car": [
        ("Maruti Suzuki", "Swift VXi", ["Petrol", "CNG"]),
        ("Maruti Suzuki", "WagonR LXi", ["Petrol", "CNG"]),
        ("Maruti Suzuki", "Baleno Zeta", ["Petrol"]),
        ("Hyundai", "i20 Sportz", ["Petrol", "Diesel"]),
        ("Hyundai", "Creta SX", ["Petrol", "Diesel"]),
        ("Tata", "Nexon XZ+", ["Petrol", "Diesel", "Electric"]),
        ("Tata", "Punch Adventure", ["Petrol"]),
        ("Mahindra", "XUV700 AX5", ["Diesel"]),
        ("Honda", "City ZX", ["Petrol"]),
        ("Toyota", "Innova Crysta", ["Diesel"]),
        ("Kia", "Seltos HTK", ["Petrol", "Diesel"]),
    ],
    "two_wheeler": [
        ("Hero", "Splendor Plus", ["Petrol"]),
        ("Honda", "Activa 6G", ["Petrol"]),
        ("TVS", "Jupiter 110", ["Petrol"]),
        ("Bajaj", "Pulsar 150", ["Petrol"]),
        ("Royal Enfield", "Classic 350", ["Petrol"]),
        ("Ola", "S1 Pro", ["Electric"]),
        ("Suzuki", "Access 125", ["Petrol"]),
    ],
    "auto": [
        ("Bajaj", "RE Compact", ["CNG", "Petrol"]),
        ("Piaggio", "Ape City", ["CNG", "Diesel"]),
        ("Mahindra", "Alfa Passenger", ["Diesel", "CNG"]),
        ("TVS", "King Duramax", ["CNG"]),
    ],
    "truck": [
        ("Tata", "1109g LPT", ["Diesel"]),
        ("Ashok Leyland", "Dost+", ["Diesel"]),
        ("Eicher", "Pro 2049", ["Diesel"]),
        ("BharatBenz", "1215RE", ["Diesel"]),
        ("Mahindra", "Bolero Pik-Up", ["Diesel"]),
    ],
    "taxi": [
        ("Maruti Suzuki", "Dzire Tour S", ["CNG", "Diesel"]),
        ("Toyota", "Etios Liva", ["Diesel"]),
        ("Hyundai", "Xcent Prime", ["CNG"]),
        ("Maruti Suzuki", "Ertiga Tour M", ["CNG"]),
    ],
    "bus": [
        ("Tata", "Starbus Ultra", ["Diesel"]),
        ("Ashok Leyland", "Viking BS6", ["Diesel"]),
        ("Eicher", "Skyline Pro", ["Diesel"]),
        ("Tata", "Marcopolo LPO", ["Diesel", "CNG"]),
    ],
}

COLOURS = ["White", "Silver", "Grey", "Black", "Blue", "Red", "Maroon",
           "Beige", "Yellow", "Green"]

# Commercial classes need fitness and permit; private ones do not. This split
# is what makes the fitness/permit rules fire only where they would in law.
COMMERCIAL = {"auto", "truck", "taxi", "bus"}

# --- where it was registered -------------------------------------------
# Real West Bengal RTO codes and their offices, plus a few neighbouring-state
# ones so the corpus is not uniformly local.
RTO_OFFICES = [
    ("WB01", "Kolkata (Beltala)", "West Bengal"),
    ("WB02", "Kolkata (Kasba)", "West Bengal"),
    ("WB06", "Kolkata (Salt Lake)", "West Bengal"),
    ("WB07", "Howrah", "West Bengal"),
    ("WB12", "Barasat (North 24 Pgs)", "West Bengal"),
    ("WB19", "Barrackpore", "West Bengal"),
    ("WB24", "Alipore (South 24 Pgs)", "West Bengal"),
    ("WB26", "Serampore (Hooghly)", "West Bengal"),
    ("WB34", "Durgapur", "West Bengal"),
    ("JH05", "Jamshedpur", "Jharkhand"),
    ("BR01", "Patna", "Bihar"),
    ("OD02", "Cuttack", "Odisha"),
    ("AS01", "Guwahati", "Assam"),
]

# Every state code the plate generator can produce, so a registration is always
# issued in the state its own number says. Without this a KA plate could be
# registered at a Kolkata RTO, which would make "registered out of state" -
# a genuine signal - indistinguishable from noise.
STATE_NAMES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman & Diu", "DL": "Delhi", "DN": "Dadra & Nagar Haveli",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu & Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra",
    "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram",
    "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha", "PB": "Punjab",
    "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand", "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

INSURERS = ["ICICI Lombard", "Bajaj Allianz", "HDFC ERGO",
            "New India Assurance", "United India Insurance",
            "Tata AIG", "Reliance General", "Oriental Insurance"]

POLICE_STATIONS = ["Lalbazar", "Jadavpur PS", "Bhowanipore PS", "Park Street PS",
                   "Shakespeare Sarani PS", "Burrabazar PS", "Alipore PS",
                   "Tollygunge PS", "Salt Lake North PS", "Howrah Sadar PS"]

# --- owners -------------------------------------------------------------
# Invented names. No real person is described by this package, and the store
# masks what it returns anyway.
FIRST_NAMES = ["Arindam", "Sudipta", "Rakesh", "Priyanka", "Ananya", "Debjani",
               "Imran", "Farhana", "Sourav", "Moumita", "Rajesh", "Kavita",
               "Subhas", "Nabanita", "Tanmoy", "Rupa", "Abhijit", "Sneha",
               "Pradip", "Ishita", "Manoj", "Sharmila"]
LAST_NAMES = ["Ghosh", "Banerjee", "Chatterjee", "Mukherjee", "Das", "Dutta",
              "Roy", "Sen", "Bose", "Sarkar", "Khan", "Ali", "Pal", "Nandi",
              "Biswas", "Mondal", "Chakraborty", "Saha"]

LOCALITIES = ["Ballygunge", "Salt Lake Sector II", "Behala", "Dum Dum",
              "Garia", "New Town", "Shyambazar", "Bhowanipore", "Jadavpur",
              "Howrah Maidan", "Tollygunge", "Kasba", "Ultadanga"]

# --- what can be pending against a vehicle ------------------------------
# (kind, statute/section, human description, typical fine range, authority)
# The trailing number is a WEIGHT, and it is load-bearing rather than flavour.
# Drawn uniformly, a quarter of everything the sentinel reported came back as
# an open criminal case: 27 of 28 "critical" findings were FIRs, and the one
# stolen vehicle actually moving through the network sat in the middle of them,
# indistinguishable. Traffic offences are ordinary and criminal cases are rare,
# so the corpus has to say so - otherwise "critical" stops carrying meaning
# exactly when it matters.
#
# (kind, section, description, fine range, authority, weight)
CASE_TYPES = [
    ("challan", "MVA s.119", "Signal jumping", (500, 1000), "traffic", 22),
    ("challan", "MVA s.183", "Over-speeding", (1000, 2000), "traffic", 20),
    ("challan", "MVA s.190(2)", "Emission norms violation", (1500, 10000), "pollution", 14),
    ("challan", "MVA s.196", "Driving uninsured", (2000, 4000), "traffic", 12),
    ("challan", "MVA s.184", "Dangerous driving", (1000, 5000), "traffic", 9),
    ("challan", "MVA s.192", "Using vehicle without permit", (5000, 10000), "transport", 6),
    ("challan", "MVA s.194", "Overloading", (20000, 40000), "transport", 5),
    ("challan", "MVA s.185", "Driving under the influence", (10000, 15000), "police", 4),
    ("summons", "MVA s.207", "Vehicle detained - papers not produced", (0, 0), "transport", 4),
    ("fir", "IPC 420", "Cheating - forged documents", (0, 0), "police", 2),
    ("fir", "IPC 379", "Theft of vehicle", (0, 0), "police", 1),
    ("fir", "NDPS s.20", "Narcotics - vehicle used in transport", (0, 0), "police", 1),
]

# Which criminal cases warrant waking someone. A narcotics case or a theft on a
# vehicle currently in traffic is a different thing from an old forgery case,
# and grading them identically is how a queue becomes wallpaper.
FIR_CRITICAL = {"IPC 379", "NDPS s.20"}

STATUSES = ["active", "stolen", "scrapped", "blacklisted", "suspended"]
