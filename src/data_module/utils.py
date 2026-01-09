def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    FCOS requires images as a list and targets as a list of dicts.
    Handles variable-sized bounding boxes across batch.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
    # return list(zip(batch)) #TODO


colors = [
    ("Red", "#D71E22"),
    ("Blue", "#1D3CE9"),
    ("Green", "#1B913E"),
    ("Pink", "#FF63D4"),
    ("Orange", "#FF8D1C"),
    ("Yellow", "#FFFF67"),
    ("Black", "#4A565E"),
    ("White", "#E9F7FF"),
    ("Purple", "#783DD2"),
    ("Brown", "#80582D"),
    ("Cyan", "#44FFF7"),
    ("Lime", "#5BFE4B"),
    ("Maroon", "#6C2B3D"),
    ("Rose", "#FFD6EC"),
    ("Banana", "#FFFFBE"),
    ("Gray", "#8397A7"),
    ("Tan", "#9F9989"),
    ("Coral", "#EC7578")
]
color_to_ind = {color: i for i, (color, _) in enumerate(colors)}
