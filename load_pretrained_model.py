
import tensorflow as tf
import config
from utils import label_map_util


def load_pretrained_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_label():
    category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)
    return category_index