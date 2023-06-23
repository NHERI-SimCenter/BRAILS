#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import requests

DRIVER_PATH = 'REPLACE-THIS-WITH-CHROMIUMDRIVER-PATH'

options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")

driver = webdriver.Chrome(options=options,executable_path=DRIVER_PATH)

#143001
data = []
for bldgid in range(1,143001):
    bldglink = f'https://skyscraperpage.com/cities/?buildingID={bldgid}'
    response = requests.get(bldglink, allow_redirects=False)

    if not(response.status_code == 200):
        continue
    print(bldgid)

    driver.get(bldglink)
    table_id = driver.find_element(By.TAG_NAME, 'tbody')
    rows = table_id.find_elements(By.TAG_NAME, "tr") # get all of the rows in the table
    
    for row in rows:
        matchFound = 'Floor Count' in row.text and 'ft' in row.text and 'Finished' in row.text
        if matchFound:
            bldgdata = row.text
            break
    if matchFound and 'United States' in bldgdata:
        bldgdata = bldgdata.split('\n')
        nfloors = int([row for row in bldgdata if 'Floor Count' in row][0].replace('Floor Count','').strip())
        bldgheight = [row for row in bldgdata if 'ft' in row][0]
        bldgheight = [int(s) for s in bldgheight.split() if s.isdigit()][0]
        consyear = [row for row in bldgdata if 'Finished' in row][0]
        consyear = [int(s) for s in consyear.split() if s.isdigit()][0]
         
        imgs = driver.find_elements_by_xpath("//img")
        for img in imgs:
            if 'maps.googleapis.com' in img.get_attribute("src"):
                location = img.get_attribute("src").split('&')
                location = [loc for loc in location if 'center' in loc][0].replace('center=','').split('%2C')
        lat = float(location[0])
        lon = float(location[1])
        data.append([lat,lon,nfloors,bldgheight,consyear])
driver.quit()


with open('tallbldgs.csv', 'w') as f:
    f.write('Latitude,Longitude,Nfloors,BldgHeight,ConsYear\n')
    for line in data:
        f.write(str(line)[1:-2]+'\n')