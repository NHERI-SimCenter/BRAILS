# Understanding NJDEP Building Inventory
There are 472,322 (453,361+18,961) buildings in this database (including 81,418 condos). This database is provided as an Arcgis gdb, 
which can also be opened by the opensource software [QGIS](https://www.qgis.org/en/site/).
Once the database is loaded in QGIS, you'll see several layers (tables). 
You can right click on each layer and export as your favorite format (json/csv/others). 

The columns in each table have been list below. 
Click on table name will lead to more details of that table, including 
* the explanation of each column
* categories of each column
* statistics of categories (numbers noted in parentheses)

## Tables

### [BuildingFootprints_Master](BuildingFootprints_Master.md)
* BldgUniqueID : Building id. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* SourceCit: Source citation
* Shape_Length : Longest edge (need to confirm with Tracy)
* Shape_Area : Area
* AttributeSourceCit : Attribute source citation

### [TBL_Basic_Attributes](TBL_Basic_Attributes.md)
* BldgUniqueID : Building id. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* HAG : Highest Adjacent Grade elevation 
* LAG : Lowest Adjacent Grade elevation 
* AreaSqFt : Area in square feet
* PamsPIN : Parcel ID from the MOD-IV
* CentroidY : Latitude
* CentroidX : Longitude 
* BldgNum : Building number
* BldgType : Building type
* Elevation : Elevation
* ElevUnits : Elevation units
* County : County code
* Municipality : Municipality 
* MunicipalityCode : Municipality code 
* BasicSourceCit : Basic attribute source citation

### [TBL_SubstantialDamageEstimator](TBL_SubstantialDamageEstimator.md)
* BldgUniqueID : Building id. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* StructureType : Structure Type Code (Residential or Non-Residential.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
* ResidenceType : Residence Type Code (Residential Type Only: Single Family Residence, Town or Row House, Manufactured House.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
* StructureUse : Structure Use Code (Non-Residential Type Only: Apartments, Commercial Retail, Mini-Warehouse, etc.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
* FoundationType : Residential Type Only: Select from domain. Not complete.
* SuperStructure : Residential Type Only: Select from domain. Not complete.
* ExteriorFinish : Residential Type Only: Select from domain. Not complete.
* ElevationLowestFloor : Typically assessor attribute. Not complete (empty).
* Story : Select from domain. Not complete.
* RoofCovering : Select from domain. Not complete.
* HVACSystem : Empty
* Quality : Empty
* NFIPCommunityID : Typically assessor attribute. 
* NFIPCommunityName : Typically assessor attribute. 
* YearConstruction : Typically assessor attribute. 
* SDESourceCit : Source citation details. 
* FloodZone : From FIRM. 
* FIRMPanelID : FIRM Panel number from S_FIRM_PAN. 

### [TBL_UDF_Hazus](TBL_UDF_Hazus.md)

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






### [CondoParcels](CondoParcels.md)
CONDOMINIUM PARCELS ATTRIBUTION
* BldgUniqueID : Building unique ID. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* PamsPIN : Municipality Code, Block, Lot, Qualifier (A qualification code that provides additional information about the parcel (parcel type, for example). Only qualifiers for condo parcels are included as part of the PamsPIN.)