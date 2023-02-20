# the purpose of this code is to generate code for feature making.
def make_feature(name,data_frame):
    print(f"{name}=np.array({data_frame}['{name}'])")
    print(f"{name}=({name}-np.mean({name})) / np.std({name})")
make_feature("TotRmsAbvGrd")
