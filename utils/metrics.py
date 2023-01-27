import torch
import shapely
from shapely.geometry import Polygon


def bbox_iou(box1, box2):
	"""
	Returns the IoU of two bounding boxes
	"""
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	nBox = box1.size()[0]

	iou = torch.zeros(nBox)
	for i in range(0, nBox):
		polygon1 = Polygon(box1[i,:].view(4,2)).convex_hull
		polygon2 = Polygon(box2[i,:].view(4,2)).convex_hull
		if polygon1.intersects(polygon2):
			try:
				inter_area = polygon1.intersection(polygon2).area
				union_area = polygon1.union(polygon2).area
				iou[i] =  inter_area / union_area
			except shapely.geos.TopologicalError:
				print('shapely.geos.TopologicalError occured, iou set to 0')
				iou[i] = 0

	return iou.to(device)

