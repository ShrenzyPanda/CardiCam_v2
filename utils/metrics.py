import torch
import torch.functional as F
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

def bbox_iou_nms(box1, box2):
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if cuda else 'cpu')

	nBox = box2.size()[0]

	iou = torch.zeros(nBox)
	polygon1 = Polygon(box1.view(4,2)).convex_hull
	for i in range(0, nBox):
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


def non_max_suppression(prediction, cls_thres=0.5, nms_thres=0.4):
	prediction = prediction.cpu()

	output = [None for _ in range(len(prediction))]

	for image_i, pred in enumerate(prediction):
		class_prob, class_pred = torch.max(F.softmax(pred[:, 9:], 1), 1)

		v = (class_prob > cls_thres).numpy()
		v = v.nonzero()

		pred = pred[v]
		class_prob = class_prob[v]
		class_pred = class_pred[v]

		# If none are remaining => process next image
		nP = pred.shape[0]
		if not nP:
			continue

		detections = torch.cat((pred[:, :9], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
		# Iterate through all predicted classes
		unique_labels = detections[:, -1].cpu().unique()
		if prediction.is_cuda:
			unique_labels = unique_labels.cuda()

		for c in unique_labels:
			detections_class = detections[detections[:, -1] == c]
			# Sort through confidence in one class
			_, conf_sort_index = torch.sort(detections_class[:, 8], descending=True)
			detections_class = detections_class[conf_sort_index]

			max_detections = []

			while detections_class.shape[0]:
				# Get detection with highest confidence and save as max detection
				max_detections.append(detections_class[0].unsqueeze(0))
				# Stop if we're at the last detection
				if len(detections_class) == 1:
					break
				# Get the IOUs for all boxes with lower confidence
				ious = bbox_iou_nms(max_detections[-1].squeeze(0)[0:8], detections_class[1:][:, 0:8])

				# Remove detections with IoU >= NMS threshold
				detections_class = detections_class[1:][ious < nms_thres]

			if len(max_detections) > 0:
				max_detections = torch.cat(max_detections).data
				# Add max detections to outputs
				output[image_i] = max_detections if output[image_i] is None else torch.cat(
					(output[image_i], max_detections))

	return output