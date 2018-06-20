# Colors for use in graphs, etc.
# RGB versions
OLIVE_GREEN_rgb = [82 / 255.0, 101 / 255.0, 91 / 255.0]
BROWN_rgb = [84 / 255.0, 62 / 255.0, 65 / 255.0]
INDIAN_RED_rgb = [205 / 255.0, 92 / 255.0, 92 / 255.0]
FIREBRICK_rgb = [178 / 255.0, 34 / 255.0, 34 / 255.0]
DARK_RED_rgb = [139 / 255.0, 0 / 255.0, 0 / 255.0]
BOYSENBERRY_rgb = [135 / 255.0, 50 / 255.0, 96 / 255.0]
FUSCHIA_rgb = [145 / 255.0, 92 / 255.0, 131 / 255.0]
QUEEN_BLUE_rgb = [67 / 255.0, 107 / 255.0, 149 / 255.0]
ORANGE_PEEL_rgb = [255 / 255.0, 159 / 255.0, 0 / 255.0]
BLACK_OLIVE_rgb = [59 / 255.0, 60 / 255.0, 54 / 255.0]
BLACK_rgb = [23 / 255.0, 23 / 255.0, 23 / 255.0]
GO_GREEN_rgb = [0 / 255.0, 171 / 255.0, 102 / 255.0]
BOTTLE_GREEN_rgb = [0 / 255.0, 106 / 255.0, 78 / 255.0]
TANGERINE_rgb = [242 / 255.0, 133 / 255.0, 0 / 255.0]
DARKEST_BLUE_rgb = [1 / 255.0, 31 / 255.0, 75 / 255.0]
DARK_BLUE_rgb = [3 / 255.0, 57 / 255.0, 108 / 255.0]
MEDIUM_BLUE_rgb = [0 / 255.0, 91 / 255.0, 150 / 255.0]
LIGHT_BLUE_rgb = [100 / 255.0, 151 / 255.0, 177 / 255.0]
LIGHTEST_BLUE_rgb = [179 / 255.0, 205 / 255.0, 224 / 255.0]


# HEX versions
INDIAN_RED_hex = "#CD5C5C"
FIREBRICK_hex = "#B22222"
DARK_RED_hex = "#8B0000"
BOYSENBERRY_hex = "#873260"
FUSCHIA_hex = "#915C83"
QUEEN_BLUE_hex = "#436B95"
ORANGE_PEEL_hex = "#FF9F00"
BLACK_OLIVE_hex = "#3B3C36"
BLACK_hex = "#171717"
GO_GREEN_hex = "#00AB66"
BOTTLE_GREEN_hex = "#006A4E"
TANGERINE_hex = "#F28500"
DARKEST_BLUE_hex = "#011f4b"
DARK_BLUE_hex = "#03396c"
MEDIUM_BLUE_hex = "#005b96"
LIGHT_BLUE_hex = "#6497b1"
LIGHTEST_BLUE_hex = "#b3cde0"

TIME_DELTA_LIST = [8.0, 12.0, 24.0]

# The distribution can be one of:
# Constant
# Beta(alpha, beta)
# Gamma(shape, scale)
# Triangular(left, mode, right)
# Uniform(low, high)
DISTRIBUTIONS = ['Constant', 'constant',
                 'Beta', 'beta',
                 'Gamma', 'gamma',
                 'Triangular', 'triangular',
                 'Uniform', 'uniform']

DIST_NAME_TO_IDX = {'Constant': 1, 'constant': 1,
                    'Beta': 2, 'beta': 2,
                    'Gamma': 3, 'gamma': 3,
                    'Triangular': 4, 'triangular': 4,
                    'Uniform': 5, 'uniform': 5}
