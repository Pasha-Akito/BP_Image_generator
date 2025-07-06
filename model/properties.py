PROPERTY_VOCAB = {
    # Basic functions
    "LEFT": 0, "RIGHT": 1, "EXISTS": 2, "EXACTLY": 3, "GREATERLA": 4,
    "MORESIMLA": 5, "GREATERLLA": 6, "MORE": 7, "EQUALNUM": 8, "GET": 9,
    "INSIDE": 13, "ALIGNED": 15, "HIGH": 16, "LOW": 17,
    # New Additions
    "CEASES": 41, # Opposite of EXISTS
    "LESSERLA": 42, # Opposite of GREATERLA
    "UNION": 47, # Together
    "DIFFNUM": 61, # almost opposite of equalnum, varying number of something
    "LESSSIMLA": 62, # Opposite of MORESIMLA
    "LESS": 63, # Opposite of MORE
    "CROSSES": 64, # something crossing something, or something crosses
    "LESSERLLA": 12, # Opposite of GREATERLLA


    # Shapes
    "CIRCLES": 22, "FIGURES": 23, "QUADRILATERALS": 24, "TRIANGLES": 25, 
    # New Additions
    "LINES": 48, #Lines are more specific figures which are not joined by the ends

    # FIGURES AND OTHER SHAPES ARE ALWAYS TERMINAL
    
    # Attributes
    "AREA": 26, "CONVEXITY": 27, "ORIENTATION": 28, "XPOS": 29, "NCORNERS": 30,
    "YPOS": 31, "COMPACTNESS": 32, "ELONGATION": 33, "DISTANCE": 34,
    "HOLES": 35, "HULLS": 36, "BIG": 18, "SMALL": 19, "SOLID": 20, "OUTLINE": 21,
    # new Additions
    "CURVATURE": 43, # How curvy something is
    "VERTICAL": 45, # How much vertical height something is
    "HORIZONTAL": 46, # How much horizontal height something is
    "AREA": 59, # The area of something    
    "ANGLES": 60, # Angles, HIGH angle means wide obtuse angle, LOW angle means sharp acute angle
    "SYMMETRY": 44, # The Symmetric properties of something, can be used with vertical, horizontal & exactly or exists
    
    # Quantifiers
    "ONE": 37, "TWO": 38, "THREE": 39, "FOUR": 40,

    # Directions (New addition)
    "LEFTD": 49, "RIGHTD": 50, "UP": 51, "DOWN": 52, # Adding D to thr end of LEFT and RIGHT to denote they are used for directions
    "MIDDLE": 53, "START": 54, "END": 55,
    "CLOCKWISE": 56, "COUNTERCLOCKWISE": 57,
    "OUTSIDE": 66,
    "INCREASING": 10, "DECREASING": 11,
    "ON": 65, # Something is on something
    "MOVING": 58 # Uses one of the above 


}