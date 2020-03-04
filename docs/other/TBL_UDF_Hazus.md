### TBL_UDF_Hazus

* BldgUniqueID    :  Building unique ID. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* Tract           :  (Complete) Value for Census Tract or an area roughly equal to a neighborhood established by the Census Bureau.
* PhoneNumber     :  (Empty) Typically assessor attribute for owner.
* OccupancyClass  :  (Incomplete) Residential, Commercial, Industrial, Agriculture, Government, Education, and Religious/Non- Profit. (BLDG_DESC, PROP_CLASS, & PROP_USE)
* BuildingType    :  (Incomplete) Core construction material type; Wood, Concrete, Steel, Masonry, Manufactured Housing. (BLDG_DESC only)
* Cost            :  (Empty) Replacement value; assessor data does not often include replacement cost. It is usually derived by considering heated or livable space and multiplied by cost per square foot.
* YearBuilt       :  (Complete) Typically assessor attribute.
* Area            :  (Complete) Heated or livable space. May or may not exist in typically assessor attributes. Can potentially be derived from building footprints.
* NumberofStories :  (Incomplete) Typically assessor attribute. Indicates number of stories. (BLDG_DESC only)
* DesignLevel     :  (Empty) Must have the year built to establish; see Hazus 
* FoundationType  :  Flood Model User Manual, Table 6.2. Flood model needs to distinguish which of seven types.
* FirstFloorHt    :  Flood model needs height (in feet) above grade. Can be based on default values assigned to foundation types or other preferred methods.
* ContentCost     :  Can be calculated from formula to be applied to final cost per Hazus Flood Model User Manual.
* BldgDmgFnID     :  (Empty) The damage function ID from Hazus would be entered in this field if anything other than the default was to be used. The damage function is based on the building characteristics defined in the items above.
* ContDmgFnID     :  (Empty) The damage function ID from Hazus would be entered in this field if anything other than the default was to be used. The damage function is based on the building characteristics defined in the items above.
* InvDmgFnID      :  (Empty) The damage function ID from Hazus would be entered in this field if anything other than the default was to be used. The damage function is based on the building characteristics defined in the items above.
* FloodProtection :  (Empty) Does protection exist, and if yes to what frequency? (e.g., 100-year). 
* ShelterCapacity :  Number of persons that can be sheltered.
* BackupPower     :  (Empty) Does backup power exist? [yes (1) or no (0)]
* Latitude        :  (Complete) Latitude of the Building Centroid (inside polygon).
* Longitude       :  (Complete) Longitude of the Building Centroid (inside polygon).
* InventoryCost   :  Can be calculated from formula to be applied to final cost per Hazus Flood Model User Manual.
* RiverineCoastal :  Can be used to identify if building is in riverine or coastal areas. Can be used to help designate the Damage Function IDs.
* HzSourceCit     :  (Complete) Source Citation details.
* Name            :  Typically assessor attribute for owner.
* Address         :  Typically assessor field for property location.
* City            :  Typically assessor field for property location – city.
* State           :  Typically assessor field for property location - state abbreviation.
* ZipCode         :  Typically assessor field for property location – Zip Code.
* Contact         :  (Empty) Typically assessor attribute for owner.
* Comment         :  String.