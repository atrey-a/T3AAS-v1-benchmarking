def create_csv(file):
    with open(file,"w") as f:
        f.write("mode,epoch,loss,acc\n")

def add_to_csv(file,mode,epoch,loss,acc):
    with open(file,"a") as f:
        f.write(f"{mode},{epoch},{loss},{acc}\n")
