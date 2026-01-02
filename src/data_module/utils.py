def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    FCOS requires images as a list and targets as a list of dicts.
    Handles variable-sized bounding boxes across batch.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
    # return list(zip(batch))


colors = [
    ("Red", "#0000FF"),
    ("Blue", "#FF0000"),
    ("Green", "#008000"),
    ("Pink", "#CBC0FF"),
    ("Orange", "#00A5FF"),
    ("Yellow", "#00FFFF"),
    ("Black", "#000000"),
    ("White", "#FFFFFF"),
    ("Purple", "#800080"),
    ("Brown", "#2A2AA5"),
    ("Cyan", "#FFFF00"),
    ("Lime", "#00FF00"),
    ("Maroon", "#000080"),
    ("Rose", "#7F00FF"),
    ("Banana", "#35E1FF"),
    ("Gray", "#808080"),
    ("Tan", "#8CB4D2"),
    ("Coral", "#807FFF"),
]
color_to_ind = {color: i for i, (color, _) in enumerate(colors)}
