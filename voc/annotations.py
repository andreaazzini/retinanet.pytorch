import os
import xml.etree.ElementTree as ET

from voc.bbox import BoundingBox
from voc.errors import UnsupportedExtensionError, UnsupportedFormatError


class AnnotationDir:
    supported_exts = ['.xml']
    supported_formats = ['voc']

    def __init__(self, path, filenames, labels, ext, fmt):
        if not ext in AnnotationDir.supported_exts:
            raise UnsupportedExtensionError(ext)
        if not fmt in AnnotationDir.supported_formats:
            raise UnsupportedFormatError(fmt)
        self.path = path
        self.filenames = ['{}{}'.format(fn, ext) for fn in filenames]
        self.labels = labels
        self.fmt = fmt
        self.ann_dict = self.build_annotations()

    def build_annotations(self):
        box_dict = {}
        if self.fmt == 'voc':
            for fn in self.filenames:
                boxes = []
                tree = ET.parse(os.path.join(self.path, fn))
                ann_tag = tree.getroot()
                
                size_tag = ann_tag.find('size')
                image_width = int(size_tag.find('width').text)
                image_height = int(size_tag.find('height').text)

                for obj_tag in ann_tag.findall('object'):
                    label = obj_tag.find('name').text

                    box_tag = obj_tag.find('bndbox')
                    left = int(box_tag.find('xmin').text)
                    top = int(box_tag.find('ymin').text)
                    right = int(box_tag.find('xmax').text)
                    bottom = int(box_tag.find('ymax').text)
                    
                    box = BoundingBox(left, top, right, bottom, image_width, image_height, self.labels.index(label))
                    boxes.append(box)
                if len(boxes) > 0:
                    box_dict[os.path.splitext(fn)[0]] = boxes
                else:
                    self.filenames.remove(fn)
        return box_dict

    def get_boxes(self, fn):
        return self.ann_dict[fn]
