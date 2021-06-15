import os
from brails.CityBuilder import CityBuilder

myAPIkey = os.environ["GoogleMapsAPIkey"]
print(myAPIkey)

cityBuilder = CityBuilder(
                  attributes=["occupancy","roofshape","softstory","elevated","year"],
                  numBldg=5,
                  #place="Miami, Florida",
                  place="Richmond, California",
                  GoogleMapAPIKey=myAPIkey
)

BIM = cityBuilder.build()

print(f"""
dir(cityBuilder):\t{dir(cityBuilder)}

BIM:\t\t{BIM}
type(BIM):\t{type(BIM)}
dir(BIM):\t{dir(BIM)}
""")

