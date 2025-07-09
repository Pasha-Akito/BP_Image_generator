PROPERTY_VOCAB = {
    # Basic functions
    "LEFT", "RIGHT", "EXISTS", "EXACTLY", 
    "GREATER", # Shapes are of more greater of some attribute than shape/part of shape
    "MORESIMILAR", # Shapes are more similar of some attirbute than shape/part of shape
    "GREATERALL", # All shapes  
    "MORE", #  
    "EQUALNUM", "GET",
    "INSIDE", "ALIGNED", "HIGH", "LOW", 
    "REGULAR", # Regular is in the context of the images or images
    
    # New Additions
    "CEASES",  # Opposite of EXISTS
    "LESSER",  # Opposite of GREATER
    "UNION",  # Together
    "DIFFNUM",  # almost opposite of equalnum, varying number of something
    "LESSSIMILAR",  # Opposite of MORESIMILAR
    "LESS",  # Opposite of MORE
    "CROSSES",  # something crossing something, or something crosses
    "LESSERALL",  # Opposite of GREATERALL
    "IDENTICAL", # Something is the same
    "OPPOSITE", # Something is the opposite

    # Shapes
    "CIRCLES", "FIGURES", "QUADRILATERALS", "TRIANGLES",
    
    # New Additions
    "LINES",  # Lines are more specific figures which are not joined by the ends

    # Attributes
    "AREA", "CONVEXITY", "CONCAVITY", "ORIENTATION", "XPOS", "NCORNERS",
    "YPOS", "COMPACTNESS", "ELONGATION", "DISTANCE",
    "HOLES", "HULLS", "BIG", "SMALL", "SOLID", "OUTLINE",
    
    # New Additions
    "CURVATURE",  # How curvy something is
    "VERTICAL",  # How much vertical height something is
    "HORIZONTAL",  # How much horizontal height something is
    "ANGLES",  # Angles, HIGH angle means wide obtuse angle, LOW angle means sharp acute angle
    "SYMMETRY",  # The Symmetric properties of something, can be used with vertical, horizontal & exactly or exists

    # Quantifiers
    "ONE", "TWO", "THREE", "FOUR",

    # Directions (New addition)
    "LEFTD", "RIGHTD", "UP", "DOWN",  # Adding D to the end of LEFT and RIGHT to denote they are used for directions
    "MIDDLE", "START", "END",
    "CLOCKWISE", "COUNTERCLOCKWISE",
    "OUTSIDE",
    "INCREASING", "DECREASING",
    "ON",  # Something is on something
    "MOVING"  # Uses one of the above
}
