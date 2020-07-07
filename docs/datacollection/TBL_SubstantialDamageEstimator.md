### TBL_SubstantialDamageEstimator
The number in the parentheses means the number of labels found in NJDEP data set.

* BldgUniqueID : Building id. The first four characters will be “NJBF,” followed by a 9-digit zero-padded number.

* RoofCovering : 
    - 5701 : Shingles - Asphalt, Wood (258,934)
    - 5702 : Clay Tiles               (54)
    - 5703 : Standing Seam (Metal)    (166)
    - 5704 : Slate                    (195)

* StructureType : Structure Type Code (Residential or Non-Residential.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5101 : Residential     (305,839)
    - 5102 : Non-Residential (49,348)
    - 5199 : Not Applicable  (270)

* ResidenceType : Residence Type Code (Residential Type Only: Single Family Residence, Town or Row House, Manufactured House.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5401 : Single Family Residence (356,460)
    - 5402 : Town or Row House       (12,742)
    - 5403 : Manufactured House      (2,862)
    - 5499 : Not Applicable          (80,608)

* StructureUse : Structure Use Code (Non-Residential Type Only: Apartments, Commercial Retail, Mini-Warehouse, etc.) (MOD-IV: BLDG_DESC, PROP_CLASS, & PROP_USE) Not complete.
    - 5501 : Apartments              (8,557)
    - 5502 : Auditorium              (10)
    - 5503 : Commercial Retail       (8,989)
    - 5504 : Convenience Store       (1,776)
    - 5505 : Courthouse              (7)
    - 5506 : Department Stores       (3)
    - 5507 : Elementary School       (538)
    - 5508 : Fast Food Restaurant    (38)
    - 5509 : Fire / Police Station   (114)
    - 5510 : Grocery Store           (10)
    - 5511 : High School             (222)
    - 5512 : Hospital                (60)
    - 5513 : Hotel                   (871)
    - 5514 : House of Worship        (1,621)
    - 5515 : Industrial              (8,553)
    - 5516 : Long-Term Care Facility (67)
    - 5517 : Mini-Warehouse          (4,298)
    - 5518 : Motel                   (220)
    - 5519 : Municipal Building      (461)
    - 5520 : Office Building         (2,173)
    - 5521 : Police Station          (9)
    - 5522 : Restaurants             (385)
    - 5523 : Strip Mall              (833)
    - 5524 : Condominium             (3,108)
    - 5588 : Other                   (15,869)
    - 5599 : Not Applicable          (393,875)

* SuperStructure : Residential Type Only. Not complete.
    - 5601 : Stud-framed    (247,143)
    - 5602 : Common Brick   (109)
    - 5603 : ICF            (221)
    - 5604 : Masonry        (46)

* ExteriorFinish : Residential Type Only. Not complete.
    - 5801 : Siding or Stucco                  (244,052)
    - 5802 : Brick Veneer                      (3,002)
    - 5803 : EIFS                              (61)
    - 5804 : None - common brick, structural   (551)

* Story : Not complete.
    - 5201 : One Story-Residential             (49,751)
    - 5202 : Two or More Stories-Residential   (194,808)
    - 5301 : 1 Story-Non-Residential           (21,011)
    - 5302 : 2 thru 4-Non-Residential          (13,567)
    - 5303 : 5 or more-Non-Residential         (277)

* FoundationType : Residential Type Only. Not complete.
    - 5001 : Countinuous Wall w/Slab         (346)
    - 5002 : Basement                        (47,136)
    - 5003 : Crawlspace                      (110,153)
    - 5004 : Piles                           (3,967)
    - 5005 : Slab - on - Grade               (86,074)
    - 5006 : Piers and Posts                 (3,325)

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

* HVACSystem : Empty

* Quality : Empty

* NFIPCommunityID : Typically assessor attribute. 

* NFIPCommunityName : Typically assessor attribute. 

* YearConstruction : Typically assessor attribute. 

* SDESourceCit : Source citation details. 

* FIRMPanelID : FIRM Panel number from S_FIRM_PAN. 

* ElevationLowestFloor : Typically assessor attribute. Not complete (empty).