from brails.InventoryGenerator import InventoryGenerator

# Initialize InventoryGenerator:
invGenerator = InventoryGenerator(location='San Rafael, CA',
                                  nbldgs=10, randomSelection=True,
                                  GoogleAPIKey="PROVIDE_YOUR_KEY")

# Run InventoryGenerator to obtain a 10-building inventory from the entered location:
# To run InventoryGenerator for all enabled attributes set attributes='all':
invGenerator.generate(attributes=['numstories','roofshape','buildingheight'])

# View generated inventory:
invGenerator.inventory
