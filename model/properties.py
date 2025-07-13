PROPERTY_VOCAB = {
    # Basic functions
    "LEFT", "RIGHT", "EXISTS", "EXACTLY", 
    "GREATER", # Shapes are of more greater of some attribute than shape/part of shape in opposite side of images
    "MORESIMILAR", # Shapes are more similar of some attirbute than shape/part of shape
    "GREATERALL", # All shapes have something greater than other shapes/part of shape in the same sides of images
    "MORE", #  
    "EQUALNUM", "GET",
    "INSIDE", "ALIGNED", "HIGH", "LOW", 
    "REGULAR", # Regular is in the context of the images or images
    
    # New Additions
    "CEASES",  # Opposite of EXISTS
    "LESSER",  # Opposite of GREATER
    "AND",  # Both conditions are statisified
    "OR", # Either condition is satisifed
    "DIFFNUM",  # almost opposite of equalnum, varying number of something
    "LESSSIMILAR",  # Opposite of MORESIMILAR
    "LESS",  # Opposite of MORE
    "LESSERALL",  # Opposite of GREATERALL
    "IDENTICAL", # Something is the same
    "OPPOSITE", # Something is the opposite

    # Shapes
    "CIRCLES", "FIGURES", "QUADRILATERALS", "TRIANGLES",
    
    # New Additions
    "LINES",  # Lines are more specific figures which are not joined by the ends

    # Attributes
    "SIZE", "CONVEXITY", "CONCAVITY", "ORIENTATION", "XPOS", "NCORNERS", "NSIDES",
    "YPOS", "COMPACTNESS", "ELONGATION", "DISTANCE",
    "HOLES", "HULLS", "BIG", "SMALL", "SOLID",
    
    # New Additions
    "VERTICAL",  # Used to get a vertical aspect of a shape/part of shape
    "HORIZONTAL",  # Used to get a horizontal aspect of shape/part of shape
    "ANGLES",  # Angles, HIGH angle means wide obtuse angle, LOW angle means sharp acute angle
    "SYMMETRY",  # The Symmetric properties of something, can be used with vertical, horizontal & exactly or exists
    "ASPECTR" # High aspect ration would equal thin and elongated, while low aspect ration equals compact

    # Quantifiers
    "ONE", "TWO", "THREE", "FOUR",

    # Directions (New addition)
    "LEFTD", "RIGHTD", "UP", "DOWN",  # Adding D to the end of LEFT and RIGHT to denote they are used for directions
    "MIDDLE", "START", "END",
    "CLOCKWISE", "COUNTERCLOCKWISE",
    "OUTWARDS",
    "INCREASING", "DECREASING",
    "ON",  # Something is on something
    "MOVING"  # Uses one of the above
}
