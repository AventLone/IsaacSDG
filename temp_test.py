pile_prim_paths = {"eu": list(), "plastic_1": list(), "plastic_2": list()}


for a, b in pile_prim_paths.items():
    for i in range(2, 7):
        b.append(i)

print(pile_prim_paths)