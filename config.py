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

img_height = 300
img_width = 300
img_channels = 3

classes = 2
class_names = ['background', 'id', 'name', 'birth', 'home', 'add', 'image']
weight_path = 'weight.h5'
data_path = 'new_data/text_annotated.txt'

labels_output_format = ('class_id', 'xmin', 'ymin', 'xmax', 'ymax')

