aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5]]
li_steps = [8, 16, 32, 64, 100, 300]
var = [0.1, 0.1, 0.2, 0.2]
subtract_mean = [123, 117, 104]
swap_channels = [2, 1, 0]

# `None` or a list with as many elements as there are predictor layers. The elements can be
#  either floats or tuples of two floats. These numbers represent for each predictor layer how many
#  pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
#  as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
#  of the step size specified in the `steps` argument. If the list contains floats, then that value will
#  be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
#  `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

#  A list of floats containing scaling factors per convolutional predictor layer. This list
#  must be one element longer than the number of predictor layers. The first 'k' elements are the scaling factors
#  for the 'k' predictor layers, while the last element is used for the second for aspect ratio 1 in the last predictor
#  layers if 'two_boxes_for_ar1' is 'True'. This additional last scaling factor must be passed either way, even if
#  it is not being used. If a list is passed, this argument overrides 'min_scale' and 'max_scale'. All scaling
#  factors must be greater than zero
scales_df = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]


img_height = 300
img_width = 300
img_channels = 3

classes = 7
class_names = ['background', 'id', 'name', 'birth', 'home', 'add', 'image']
weight_path = 'weight.h5'
data_path = 'new_data/text_annotated.txt'

labels_output_format = ('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
