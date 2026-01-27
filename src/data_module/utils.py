def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    FCOS requires images as a list and targets as a list of dicts.
    Handles variable-sized bounding boxes across batch.
    """
    
    return {key:[elem[key] for elem in batch] for key in batch[0]}


colors = [
    ("red", "#D71E22"),
    ("blue", "#1D3CE9"),
    ("green", "#1B913E"),
    ("pink", "#FF63D4"),
    ("orange", "#FF8D1C"),
    ("yellow", "#FFFF67"),
    ("black", "#4A565E"),
    ("white", "#E9F7FF"),
    ("purple", "#783DD2"),
    ("brown", "#80582D"),
    ("cyan", "#44FFF7"),
    ("lime", "#5BFE4B"),
    ("maroon", "#6C2B3D"),
    ("rose", "#FFD6EC"),
    ("banana", "#FFFFBE"),
    ("gray", "#8397A7"),
    ("tan", "#9F9989"),
    ("coral", "#EC7578")
]
color_to_ind = {color: i for i, (color, _) in enumerate(colors)}
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)
