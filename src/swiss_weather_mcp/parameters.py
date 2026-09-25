"""
The parameters the server reads from the MeteoSwiss local forecast collection.

Everything here changes only when MeteoSwiss changes what it publishes. The comment on each code
is the MeteoSwiss description of that parameter, from ogd-local-forecasting_meta_parameters.csv.

Hourly times are UTC. An hourly average or sum is stamped at the end of the hour it covers, and a
3 hourly one at the end of its 3 hours. The SNAPSHOTS below are values at the moment of their stamp
instead. A daily value is not a UTC time: its stamp, such as 202609230000, is the Swiss calendar day
it describes, 00:00 to 24:00 Swiss local time.

Temperature and rain are the median of the forecast unless they are named as a quantile: half of
the possible outcomes lie below it and half above, which is not the same as the most likely value.
Medians do not add up: hourly rain medians can all be 0 on a day whose median total is not.
"""

# ── hourly ───────────────────────────────────────────────────────────────────
TEMPERATURE = "tre200h0"                # Air temperature 2 m above ground; hourly mean [°C]
WIND_SPEED = "fu3010h0"                 # Wind speed scalar; hourly mean in km/h
WIND_GUSTS = "fu3010h1"                 # Gust peak (one second); hourly maximum in km/h
WIND_DIRECTION = "dkl010h0"             # Wind direction; hourly mean [°]
PRECIPITATION = "rre150h0"              # Precipitation; hourly total [mm]
PRECIPITATION_PROBABILITY = "rp0003i0"  # Probability of precipitation during 3 hours [%]
PRECIPITATION_3H = "rre003i0"           # Total precipitation during 3 hours [mm]
PRECIPITATION_Q90 = "rreq90h0"          # Precipitation; hourly total, 90% quantile [mm]
SUNSHINE = "sre000h0"                   # Sunshine duration; hourly total [min]
FREEZING_LEVEL = "zprfr0hs"             # Zero degree level; hourly value, forecast [m]
CLOUD_COVER_LOW = "nprolohs"            # Low cloud cover [fraction 0..1]
CLOUD_COVER_MEDIUM = "npromths"         # Medium cloud cover [fraction 0..1]
CLOUD_COVER_HIGH = "nprohihs"           # High cloud cover [fraction 0..1]
WEATHER_PICTOGRAM = "jww003i0"          # MeteoSwiss-Icon, weathertype, preceding 3 hours, forecast [code]

# Hourly values taken at the moment of their stamp rather than over the hour before it. MeteoSwiss:
# "in some specific cases an instantaneous value (cloud cover, zero degree level)".
SNAPSHOTS = {FREEZING_LEVEL, CLOUD_COVER_LOW, CLOUD_COVER_MEDIUM, CLOUD_COVER_HIGH}

# ── daily, 00:00 - 24:00 Swiss local time ────────────────────────────────────
TEMPERATURE_DAY_MIN = "tre200pn"        # Air temperature 2 m above ground; daily minimum [°C]
TEMPERATURE_DAY_MAX = "tre200px"        # Air temperature 2 m above ground; daily maximum [°C]
PRECIPITATION_DAY = "rka150p0"          # Precipitation; daily total [mm]
PRECIPITATION_DAY_Q10 = "rreq10p0"      # Precipitation; daily total, 10% quantile [mm]
PRECIPITATION_DAY_Q90 = "rreq90p0"      # Precipitation; daily total, 90% quantile [mm]
WEATHER_PICTOGRAM_DAY = "jp2000d0"      # MeteoSwiss pictogram number, daily value (valid for daytime period) [code]

# MeteoSwiss pictogram codes, published by jp2000d0 (daily) and jww003i0 (3 hourly). Codes above 100
# are the night variant of the same weather. Each code has its description, taken from the MeteoSwiss
# icon reference sheet with its "cloudly" and "intermittant" spellings corrected because these strings
# are shown to the user, and a standard Unicode emoji, since MeteoSwiss's own pictogram images are
# its own design. The emoji shows the kind of weather, not its strength.
PICTOGRAMS = {
    1: ("sunny", "☀️"),
    2: ("mostly sunny, some clouds", "🌤️"),
    3: ("partly sunny, thick passing clouds", "⛅"),
    4: ("overcast", "☁️"),
    5: ("very cloudy", "🌥️"),
    6: ("sunny intervals, isolated showers", "🌦️"),
    7: ("sunny intervals, isolated sleet", "🌨️"),
    8: ("sunny intervals, snow showers", "🌨️"),
    9: ("overcast, some rain showers", "🌧️"),
    10: ("overcast, some sleet", "🌨️"),
    11: ("overcast, some snow showers", "🌨️"),
    12: ("sunny intervals, chance of thunderstorms", "⛈️"),
    13: ("sunny intervals, possible thunderstorms", "⛈️"),
    14: ("very cloudy, light rain", "🌧️"),
    15: ("very cloudy, light sleet", "🌨️"),
    16: ("very cloudy, light snow showers", "🌨️"),
    17: ("very cloudy, intermittent rain", "🌧️"),
    18: ("very cloudy, intermittent sleet", "🌨️"),
    19: ("very cloudy, intermittent snow", "🌨️"),
    20: ("very overcast with rain", "🌧️"),
    21: ("very overcast with frequent sleet", "🌨️"),
    22: ("very overcast with heavy snow", "🌨️"),
    23: ("very overcast, slight chance of storms", "⛈️"),
    24: ("very overcast with storms", "⛈️"),
    25: ("very cloudy, very stormy", "⛈️"),
    26: ("high clouds", "🌤️"),
    27: ("stratus", "☁️"),
    28: ("fog", "🌫️"),
    29: ("sunny intervals, scattered showers", "🌦️"),
    30: ("sunny intervals, scattered snow showers", "🌨️"),
    31: ("sunny intervals, scattered sleet", "🌨️"),
    32: ("sunny intervals, some showers", "🌦️"),
    33: ("short sunny intervals, frequent rain", "🌦️"),
    34: ("short sunny intervals, frequent snowfalls", "🌨️"),
    35: ("overcast and dry", "☁️"),
    36: ("partly sunny, slightly stormy", "⛈️"),
    37: ("partly sunny, stormy snow showers", "⛈️"),
    38: ("overcast, thundery showers", "⛈️"),
    39: ("overcast, thundery snow showers", "⛈️"),
    40: ("very cloudy, slightly stormy", "⛈️"),
    41: ("overcast, slightly stormy", "⛈️"),
    42: ("very cloudy, thundery snow showers", "⛈️"),
    101: ("clear", "✨"),
    102: ("slightly overcast", "🌙"),
    103: ("heavy cloud formations", "☁️"),
    104: ("overcast", "☁️"),
    105: ("very cloudy", "☁️"),
    106: ("overcast, scattered showers", "🌧️"),
    107: ("overcast, scattered rain and snow showers", "🌨️"),
    108: ("overcast, snow showers", "🌨️"),
    109: ("overcast, some showers", "🌧️"),
    110: ("overcast, some rain and snow showers", "🌨️"),
    111: ("overcast, some snow showers", "🌨️"),
    112: ("slightly stormy", "⛈️"),
    113: ("storms", "⛈️"),
    114: ("very cloudy, light rain", "🌧️"),
    115: ("very cloudy, light rain and snow showers", "🌨️"),
    116: ("very cloudy, light snowfall", "🌨️"),
    117: ("very cloudy, intermittent rain", "🌧️"),
    118: ("very cloudy, intermittent mixed rain and snowfall", "🌨️"),
    119: ("very cloudy, intermittent snowfall", "🌨️"),
    120: ("very cloudy, constant rain", "🌧️"),
    121: ("very cloudy, frequent rain and snowfall", "🌨️"),
    122: ("very cloudy, heavy snowfall", "🌨️"),
    123: ("very cloudy, slightly stormy", "⛈️"),
    124: ("very cloudy, stormy", "⛈️"),
    125: ("very cloudy, storms", "⛈️"),
    126: ("high cloud", "🌙"),
    127: ("stratus", "☁️"),
    128: ("fog", "🌫️"),
    129: ("slightly overcast, scattered showers", "🌧️"),
    130: ("slightly overcast, scattered snowfall", "🌨️"),
    131: ("slightly overcast, rain and snow showers", "🌨️"),
    132: ("slightly overcast, some showers", "🌧️"),
    133: ("overcast, frequent snow showers", "🌨️"),
    134: ("overcast, frequent snow showers", "🌨️"),
    135: ("overcast and dry", "☁️"),
    136: ("slightly overcast, slightly stormy", "⛈️"),
    137: ("slightly overcast, stormy snow showers", "⛈️"),
    138: ("overcast, thundery showers", "⛈️"),
    139: ("overcast, thundery snow showers", "⛈️"),
    140: ("very cloudy, slightly stormy", "⛈️"),
    141: ("overcast, slightly stormy", "⛈️"),
    142: ("very cloudy, thundery snow showers", "⛈️"),
}
