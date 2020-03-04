### TBL_SubstantialDamageEstimator
* BldgUniqueID : Building id. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.
* StructureType : Structure Type Code (Residential or Non-Residential.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5101 : Residential
    - 5102 : Non-Residential
    - 5199 : Not Applicable
* ResidenceType : Residence Type Code (Residential Type Only: Single Family Residence, Town or Row House, Manufactured House.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5401 : Single Family Residence
    - 5402 : Town or Row House
    - 5403 : Manufactured House
    - 5499 : Not Applicable
* StructureUse : Structure Use Code (Non-Residential Type Only: Apartments, Commercial Retail, Mini-Warehouse, etc.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5501 : Apartments
    - 5502 : Auditorium
    - 5503 : Commercial Retail
    - 5504 : Convenience Store
    - 5505 : Courthouse
    - 5506 : Department Stores
    - 5507 : Elementary School
    - 5508 : Fast Food Restaurant
    - 5509 : Fire / Police Station
    - 5510 : Grocery Store
    - 5511 : High School
    - 5512 : Hospital
    - 5513 : Hotel
    - 5514 : House of Worship
    - 5515 : Industrial
    - 5516 : Long-Term Care Facility
    - 5517 : Mini-Warehouse
    - 5518 : Motel
    - 5519 : Municipal Building
    - 5520 : Office Building
    - 5521 : Police Station
    - 5522 : Restaurants
    - 5523 : Strip Mall
    - 5524 : Condominium
    - 5588 : Other
    - 5599 : Not Applicable
* FoundationType : Residential Type Only. Not complete.
    - 5001 : Countinuous Wall w/Slab
    - 5002 : Basement
    - 5003 : Crawlspace
    - 5004 : Piles
    - 5005 : Slab - on - Grade
    - 5006 : Piers and Posts
* SuperStructure : Residential Type Only. Not complete.
    - 5601 : Stud-framed
    - 5602 : Common Brick
    - 5603 : ICF
    - 5604 : Masonry
* ExteriorFinish : Residential Type Only. Not complete.
    - 5801 : Siding or Stucco
    - 5802 : Brick Veneer
    - 5803 : EIFS
    - 5804 : None - common brick, structural
* ElevationLowestFloor : Typically assessor attribute. Not complete (empty).
* Story : Not complete.
    - 5201 : One Story-Residential
    - 5202 : Two or More Stories-Residential
    - 5301 : 1 Story-Non-Residential
    - 5302 : 2 thru 4-Non-Residential
    - 5303 : 5 or more-Non-Residential
* RoofCovering : 
    - 5701 : Shingles - Asphalt, Wood
    - 5702 : Clay Tiles
    - 5703 : Standing Seam (Metal)
    - 5704 : Slate
* HVACSystem : Empty
* Quality : Empty
* NFIPCommunityID : Typically assessor attribute. 
* NFIPCommunityName : Typically assessor attribute. 
* YearConstruction : Typically assessor attribute. 
* SDESourceCit : Source citation details. 
* FloodZone : From FIRM. 
    - 6101 : VE - RIVERINE FLOODWAY SHOWN IN COASTAL ZONE
    - 6102 : VE
    - 6103 : AE - FLOODWAY
    - 6104 : AE - ADMINISTRATIVE FLOODWAY
    - 6105 : AO - FLOODWAY
    - 6106 : AE
    - 6107 : AH
    - 6108 : AO
    - 6109 : A
    - 6110 : X - 0.2 PCT ANNUAL CHANCE FLOOD HAZARD
    - 6111 : X - AREA WITH REDUCED FLOOD RISK DUE TO LEVEE
    - 6112 : X - AREA OF MINIMAL FLOOD HAZARD
    - 6113 : OPEN WATER
    - 6114 : D
    - 6115 : AREA NOT INCLUDED
    - 6199 : NO ZONE AVAILABLE
* FIRMPanelID : FIRM Panel number from S_FIRM_PAN. 