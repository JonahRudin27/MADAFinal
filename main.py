"Region 1: Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, "
"Vermont; "
"Region 2: New Jersey, New York; "
"Region 3: Delaware, District of Columbia, "
"Maryland, Pennsylvania, Virginia, West Virginia; "
"Region 4: Alabama, Florida, "
"Georgia, Kentucky, Mississippi, North Carolina, South Carolina, Tennessee; "
"Region 5: Illinois, Indiana, Michigan, Minnesota, Ohio, Wisconsin; "
"Region 6: Arkansas, Louisiana, New Mexico, Oklahoma, Texas; "
"Region 7: Iowa, Kansas, Missouri, Nebraska; "
"Region 8: Colorado, Montana, North Dakota, South Dakota, Utah, Wyoming; "
"Region 9: Arizona, California, Hawaii, Nevada; Region 10: Alaska, Idaho, Oregon, Washington."

from model import Model

model = Model(True)
model.test_models()

for feature in model.poly_feature_names:
    model.graph_feature(feature)