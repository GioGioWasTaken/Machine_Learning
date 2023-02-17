# the purpose of this code is to generate code for feature making.
def make_feature(name):
    print(f"{name}=np.array(data['{name}'])")
    print(f"{name}=({name}-np.mean({name})) / np.std({name})")
make_feature("TotRmsAbvGrd")
# POSSIBLE FEATURES:  GarageQual, GarageArea, GarageCars WoodDeckSF, TotRmsAbvGrd, BedroomAbvGr, GrLivArea
