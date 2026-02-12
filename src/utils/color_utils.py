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
    ("coral", "#EC7578"),
]
color_to_ind = {color: i for i, (color, _) in enumerate(colors)}


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def distance_bgr(bgr1, bgr2):
    return sum(abs(x - y) for x, y in zip(bgr1, bgr2))
